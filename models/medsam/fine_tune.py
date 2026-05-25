#!/usr/bin/env python3
"""
Fine-tune MedSAM on CHD benchmark (fold 0 train/val split).

Strategy:
  - Freeze image encoder (ViT-B), train mask decoder + prompt encoder
  - Input: axial 2D slices sampled from training volumes
  - Prompt: ground-truth bounding box with ±N-pixel jitter (data augmentation)
  - Loss: binary cross-entropy + Dice per class (one-vs-rest)
  - One model per class (7 binary models) or single multi-label model

This script trains the single-class variant (recommended for CHD structures).

Usage:
    python models/medsam/fine_tune.py \
        --splits-json chd_benchmark/splits/holdout.json \
        --images-dir data/imagesTr \
        --labels-dir data/labelsTr \
        --checkpoint models/medsam/checkpoints/medsam_vit_b.pth \
        --output-dir models/medsam/checkpoints/finetuned \
        --class-id 1          # train one class at a time (1-7)

After training all classes, run inference:
    python models/medsam/run_finetuned.py ...

Environment: environments/medsam.yml
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


LABEL_NAMES = {
    1: "left_ventricle", 2: "right_ventricle", 3: "left_atrium",
    4: "right_atrium", 5: "myocardium", 6: "aorta", 7: "pulmonary_artery",
}


def bbox_from_mask(mask: np.ndarray, jitter: int = 10) -> list[int]:
    """Return [x1, y1, x2, y2] bounding box from a binary mask with jitter."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, mask.shape[1], mask.shape[0]]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = mask.shape
    rmin = max(0, rmin - random.randint(0, jitter))
    rmax = min(H - 1, rmax + random.randint(0, jitter))
    cmin = max(0, cmin - random.randint(0, jitter))
    cmax = min(W - 1, cmax + random.randint(0, jitter))
    return [cmin, rmin, cmax, rmax]  # x1, y1, x2, y2


def normalize_slice(sl: np.ndarray) -> np.ndarray:
    lo, hi = sl.min(), sl.max()
    if hi == lo:
        return np.zeros_like(sl, dtype=np.float32)
    return ((sl - lo) / (hi - lo)).astype(np.float32)


class SAMSliceDataset(Dataset):
    def __init__(self, case_ids: list[str], images_dir: Path, labels_dir: Path, class_id: int):
        self.slices: list[tuple[Path, Path, int]] = []  # (img_path, lbl_path, z)
        self.class_id = class_id

        for case_id in case_ids:
            img_candidates = list(images_dir.glob(f"{case_id}_0000.nii.gz")) + list(images_dir.glob(f"{case_id}.nii.gz"))
            lbl_candidates = list(labels_dir.glob(f"{case_id}.nii.gz"))
            if not img_candidates or not lbl_candidates:
                continue
            img_path, lbl_path = img_candidates[0], lbl_candidates[0]
            lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)
            # Only include slices where this class is present
            for z in range(lbl.shape[2]):
                if (lbl[:, :, z] == class_id).any():
                    self.slices.append((img_path, lbl_path, z))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, lbl_path, z = self.slices[idx]
        img = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
        lbl = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.uint8)

        sl = normalize_slice(img[:, :, z])
        mask = (lbl[:, :, z] == self.class_id).astype(np.float32)

        # Resize to 1024x1024
        from skimage.transform import resize
        sl_r = resize(sl, (1024, 1024), order=1, preserve_range=True).astype(np.float32)
        mask_r = resize(mask, (256, 256), order=0, preserve_range=True).astype(np.float32)

        image_tensor = torch.from_numpy(np.stack([sl_r, sl_r, sl_r], axis=0))  # 3 x 1024 x 1024

        # Bounding box prompt from resized mask (undo mask resize to get bbox)
        mask_1024 = resize(mask, (1024, 1024), order=0, preserve_range=True).astype(np.float32)
        bbox = bbox_from_mask(mask_1024 > 0.5)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return image_tensor, bbox_tensor, torch.from_numpy(mask_r)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - (2 * intersection + eps) / (union + eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-json", type=Path, default=Path("chd_benchmark/splits/holdout.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/imagesTr"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labelsTr"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/medsam/checkpoints/medsam_vit_b.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/medsam/checkpoints/finetuned"))
    parser.add_argument("--class-id", type=int, required=True, choices=range(1, 8))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.splits_json) as f:
        splits = json.load(f)
    train_ids = splits["train"]

    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    model = model.to(args.device)

    # Freeze image encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    dataset = SAMSliceDataset(train_ids, args.images_dir, args.labels_dir, args.class_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"Fine-tuning MedSAM for class {args.class_id} ({LABEL_NAMES[args.class_id]}) "
          f"on {len(dataset)} slices, {args.epochs} epochs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / f"medsam_chd_class{args.class_id}.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.image_encoder.eval()  # keep BN in eval for frozen encoder
        total_loss = 0.0
        for images, bboxes, masks in loader:
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            masks = masks.to(args.device).unsqueeze(1)  # B x 1 x H x W

            with torch.no_grad():
                image_embeddings = model.image_encoder(model.preprocess(images))
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=bboxes.unsqueeze(1),
                masks=None,
            )
            pred_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )[:2]

            loss = F.binary_cross_entropy_with_logits(pred_masks, masks) + dice_loss(pred_masks, masks).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch}/{args.epochs}  loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
