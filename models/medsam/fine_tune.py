#!/usr/bin/env python3
"""
MedSAM2 fine-tune on CHD benchmark — one class at a time.

SAM2 architecture differences from SAM1:
  - image_encoder returns multi-scale feature dicts, not a single tensor
  - prompt encoder is model.sam_prompt_encoder (not model.prompt_encoder)
  - mask decoder is model.sam_mask_decoder (not model.mask_decoder)
  - no model.preprocess() — preprocessing is in SAM2ImagePredictor

Strategy: use predictor.set_image() to encode each image (encoder frozen,
no_grad OK), then run sam_prompt_encoder + sam_mask_decoder with gradients
to train the mask decoder.

Usage:
    python models/medsam/fine_tune.py \
        --splits-json   chd_benchmark/splits/holdout.json \
        --images-dir    data/imagesTr \
        --labels-dir    data/labelsTr \
        --medsam2-dir   models/medsam/MedSAM2 \
        --checkpoint    models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt \
        --output-dir    models/medsam/checkpoints/finetuned \
        --class-id      1

Environment: medsam2_env
"""

import argparse
import json
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LABEL_NAMES = {
    1: "left_ventricle", 2: "right_ventricle", 3: "left_atrium",
    4: "right_atrium",   5: "myocardium",       6: "aorta",
    7: "pulmonary_artery",
}


def normalize_to_uint8(volume: np.ndarray, wc: int = 40, ww: int = 400) -> np.ndarray:
    lo, hi = wc - ww // 2, wc + ww // 2
    vol = np.clip(volume, lo, hi)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, jitter: int = 10) -> list[int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        H, W = mask.shape
        return [0, 0, W, H]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = mask.shape
    return [
        max(0, int(cmin) - random.randint(0, jitter)),
        max(0, int(rmin) - random.randint(0, jitter)),
        min(W - 1, int(cmax) + random.randint(0, jitter)),
        min(H - 1, int(rmax) + random.randint(0, jitter)),
    ]


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - (2 * inter + eps) / (union + eps)


class CHDSliceDataset(Dataset):
    """Returns (image_rgb_np, bbox_np, mask_128_tensor) per slice."""

    def __init__(self, case_ids: list, images_dir: Path, labels_dir: Path, class_id: int):
        self.class_id = class_id
        self.slices: list[tuple] = []

        for case_id in case_ids:
            img_cands = (list(images_dir.glob(f"{case_id}_0000.nii.gz")) +
                         list(images_dir.glob(f"{case_id}.nii.gz")))
            lbl_cands = list(labels_dir.glob(f"{case_id}.nii.gz"))
            if not img_cands or not lbl_cands:
                continue
            lbl = np.asarray(nib.load(str(lbl_cands[0])).dataobj, dtype=np.uint8)
            for z in range(lbl.shape[2]):
                if (lbl[:, :, z] == class_id).any():
                    self.slices.append((img_cands[0], lbl_cands[0], z))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        from skimage.transform import resize
        img_path, lbl_path, z = self.slices[idx]
        img = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
        lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)

        sl = normalize_to_uint8(img[:, :, z])
        mask = (lbl[:, :, z] == self.class_id).astype(np.float32)

        # SAM2 expects 1024x1024 RGB uint8
        sl_1024 = resize(sl, (1024, 1024), order=1, preserve_range=True).astype(np.uint8)
        sl_rgb = np.stack([sl_1024, sl_1024, sl_1024], axis=-1)  # (1024, 1024, 3)

        # GT bbox with jitter (on 1024 scale)
        mask_1024 = resize(mask, (1024, 1024), order=0, preserve_range=True) > 0.5
        bbox = np.array(bbox_from_mask(mask_1024), dtype=np.float32)

        # Target mask at 128x128 — MedSAM2 tiny (hiera_t512) decoder outputs 128x128
        mask_128 = torch.from_numpy(
            resize(mask, (128, 128), order=0, preserve_range=True).astype(np.float32)
        ).unsqueeze(0)  # (1, 128, 128)

        return sl_rgb, bbox, mask_128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-json",  type=Path, default=Path("chd_benchmark/splits/holdout.json"))
    parser.add_argument("--images-dir",   type=Path, default=Path("data/imagesTr"))
    parser.add_argument("--labels-dir",   type=Path, default=Path("data/labelsTr"))
    parser.add_argument("--medsam2-dir",  type=Path, default=Path("models/medsam/MedSAM2"))
    parser.add_argument("--checkpoint",   type=Path, default=Path("models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt"))
    parser.add_argument("--config",       default="configs/sam2.1_hiera_t512")
    parser.add_argument("--output-dir",   type=Path, default=Path("models/medsam/checkpoints/finetuned"))
    parser.add_argument("--class-id",     type=int, required=True, choices=range(1, 8))
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.medsam2_dir))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    with open(args.splits_json) as f:
        splits = json.load(f)

    dataset = CHDSliceDataset(splits["train"], args.images_dir, args.labels_dir, args.class_id)
    # DataLoader with batch_size=1 — we use the predictor per image
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    model = build_sam2(args.config, str(args.checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)

    # Freeze image encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    cls_name = LABEL_NAMES[args.class_id]
    print(f"Fine-tuning MedSAM2 for class {args.class_id} ({cls_name}): "
          f"{len(dataset)} slices, {args.epochs} epochs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"medsam2_chd_class{args.class_id}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.image_encoder.eval()
        total_loss = 0.0

        for img_rgb, bbox, mask_128 in loader:
            # img_rgb: (1, 1024, 1024, 3) uint8 numpy-style tensor
            # bbox: (1, 4) float32
            # mask_128: (1, 1, 128, 128) float32

            img_np = img_rgb[0].numpy()   # (1024, 1024, 3) uint8
            bbox_t = bbox.to(args.device)  # (1, 4)
            mask_t = mask_128.to(args.device)  # (1, 1, 128, 128)

            # Encode image using predictor (frozen encoder, no_grad internally)
            predictor.set_image(img_np)
            image_embed  = predictor._features["image_embed"]   # (1, C, H, W)
            high_res_feats = predictor._features["high_res_feats"]  # list of (1, C, H, W)

            # Prompt encoder — runs with gradients
            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=None,
                boxes=bbox_t.unsqueeze(1),  # (1, 1, 4)
                masks=None,
            )

            # Mask decoder — runs with gradients
            pred_masks, _, _, _ = model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats,
            )
            # pred_masks: (1, 1, 128, 128) — MedSAM2 tiny decoder output size

            loss = (F.binary_cross_entropy_with_logits(pred_masks, mask_t)
                    + dice_loss(pred_masks, mask_t).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch}/{args.epochs}  loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
