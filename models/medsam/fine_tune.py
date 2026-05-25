#!/usr/bin/env python3
"""
MedSAM2 fine-tune on CHD benchmark — one class at a time.

Freezes the image encoder (ViT), trains the mask decoder on CHD training slices.
Uses ground-truth bounding boxes (with jitter) as prompts.

Usage:
    python models/medsam/fine_tune.py \
        --splits-json   chd_benchmark/splits/holdout.json \
        --images-dir    data/imagesTr \
        --labels-dir    data/labelsTr \
        --medsam2-dir   models/medsam/MedSAM2 \
        --checkpoint    models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt \
        --output-dir    models/medsam/checkpoints/finetuned \
        --class-id      1   # run once per class (1-7); SLURM job loops over all

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
    ]  # x1, y1, x2, y2


class CHDSliceDataset(Dataset):
    def __init__(self, case_ids: list, images_dir: Path, labels_dir: Path, class_id: int):
        self.class_id = class_id
        self.slices: list[tuple] = []  # (img_path, lbl_path, z)

        for case_id in case_ids:
            img_cands = list(images_dir.glob(f"{case_id}_0000.nii.gz")) + \
                        list(images_dir.glob(f"{case_id}.nii.gz"))
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
        img_path, lbl_path, z = self.slices[idx]
        img = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
        lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)

        sl = normalize_to_uint8(img[:, :, z])
        mask = (lbl[:, :, z] == self.class_id).astype(np.float32)

        # SAM2 expects 1024x1024 RGB
        from skimage.transform import resize
        sl_r = resize(sl, (1024, 1024), order=1, preserve_range=True).astype(np.uint8)
        mask_r = resize(mask, (256, 256), order=0, preserve_range=True).astype(np.float32)

        # GT bbox with jitter on 1024-scale mask
        mask_1024 = resize(mask, (1024, 1024), order=0, preserve_range=True) > 0.5
        bbox = bbox_from_mask(mask_1024)

        image_t = torch.from_numpy(
            np.stack([sl_r, sl_r, sl_r], axis=0).astype(np.float32)
        )  # 3 x 1024 x 1024
        bbox_t = torch.tensor(bbox, dtype=torch.float32)
        mask_t = torch.from_numpy(mask_r).unsqueeze(0)  # 1 x 256 x 256

        return image_t, bbox_t, mask_t


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - (2 * inter + eps) / (union + eps)


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
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(args.medsam2_dir))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    with open(args.splits_json) as f:
        splits = json.load(f)

    dataset = CHDSliceDataset(splits["train"], args.images_dir, args.labels_dir, args.class_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = build_sam2(args.config, str(args.checkpoint), device=args.device)

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

        for images, bboxes, masks in loader:
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            masks  = masks.to(args.device)

            with torch.no_grad():
                image_embeddings = model.image_encoder(model.preprocess(images))

            sparse_emb, dense_emb = model.sam_prompt_encoder(
                points=None,
                boxes=bboxes.unsqueeze(1),
                masks=None,
            )
            pred_masks, _ = model.sam_mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None,
            )

            loss = F.binary_cross_entropy_with_logits(pred_masks, masks) \
                 + dice_loss(pred_masks, masks).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch}/{args.epochs}  loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
