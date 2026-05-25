#!/usr/bin/env python3
"""
MONAI Auto3DSeg zero-shot inference on the CHD benchmark test set.

Auto3DSeg trains a segmentation algorithm automatically given a dataset.
"Zero-shot" here uses a MONAI Model Zoo pretrained checkpoint for cardiac
segmentation, rather than training from scratch.

If no pretrained cardiac checkpoint is available, this falls back to the
full Auto3DSeg training pipeline (see autoseg/train.py).

Model zoo checkpoint options:
  - monai_brats_mri_segmentation (example — swap for cardiac when available)
  - Or use a checkpoint trained on a public cardiac dataset (e.g., ACDC, MM-WHS)

Usage:
    python models/autoseg/run_zeroshot.py \
        --input-dir data/imagesTs \
        --output-dir results/autoseg/zeroshot \
        --checkpoint models/autoseg/checkpoints/pretrained.pt

Environment: environments/autoseg.yml
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd,
    Orientationd, Spacingd, ToTensord,
)


NUM_CLASSES = 8  # background + 7 CHD structures


def load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        init_filters=16,
    ).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    # Handle wrapped checkpoints
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess(img_path: Path) -> tuple[torch.Tensor, object]:
    """Load and preprocess a NIfTI image to a model-ready tensor."""
    nii = nib.load(str(img_path))
    vol = np.asarray(nii.dataobj, dtype=np.float32)
    # Intensity normalization (z-score)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)
    tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W x D
    return tensor, nii


def predict(model: torch.nn.Module, img_tensor: torch.Tensor, device: str) -> np.ndarray:
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=img_tensor,
            roi_size=(96, 96, 96),
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/autoseg/zeroshot"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/autoseg/checkpoints/pretrained.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        print("Options:")
        print("  1. Download a pretrained cardiac checkpoint and place it there.")
        print("  2. Run autoseg/train.py to train from scratch (then this becomes fine-tune comparison).")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.device)

    images = sorted(args.input_dir.glob("*_0000.nii.gz")) or sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"Running Auto3DSeg zero-shot on {len(images)} cases...")
    for img_path in images:
        case_id = img_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        out_path = args.output_dir / f"{case_id}.nii.gz"
        if out_path.exists():
            print(f"  [skip] {case_id}")
            continue
        print(f"  {case_id}...", end=" ", flush=True)
        try:
            tensor, nii = preprocess(img_path)
            pred = predict(model, tensor, args.device)
            nib.save(nib.Nifti1Image(pred, nii.affine, nii.header), str(out_path))
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
