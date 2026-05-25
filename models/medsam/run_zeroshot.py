#!/usr/bin/env python3
"""
MedSAM2 zero-shot inference on CHD test set (imagesTs).

Uses SAM2ImagePredictor with a coarse cardiac bounding-box prior per axial slice.
Runs each CHD structure class independently using anatomically-derived bbox priors.

Usage:
    python models/medsam/run_zeroshot.py \
        --input-dir data/imagesTs \
        --output-dir results/medsam2/zeroshot \
        --medsam2-dir models/medsam/MedSAM2 \
        --checkpoint models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt

Environment: medsam2_env
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


LABEL_NAMES = {
    1: "left_ventricle",
    2: "right_ventricle",
    3: "left_atrium",
    4: "right_atrium",
    5: "myocardium",
    6: "aorta",
    7: "pulmonary_artery",
}

# Coarse anatomical bbox priors as fraction of (H, W) for axial slices.
# These are approximate centers for each structure in a typical chest CT.
# Format: (x1_frac, y1_frac, x2_frac, y2_frac)
CARDIAC_BBOX_PRIORS: dict[int, tuple] = {
    1: (0.35, 0.35, 0.60, 0.65),   # LV — central-left
    2: (0.45, 0.30, 0.70, 0.60),   # RV — central-right/anterior
    3: (0.30, 0.40, 0.55, 0.65),   # LA — posterior-left
    4: (0.45, 0.35, 0.70, 0.65),   # RA — posterior-right
    5: (0.30, 0.30, 0.65, 0.65),   # Myocardium — wraps ventricles
    6: (0.40, 0.25, 0.60, 0.55),   # Aorta — central/superior
    7: (0.40, 0.20, 0.65, 0.50),   # PA — anterior/superior
}


def normalize_to_uint8(volume: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
    """CT windowing to [0,255] uint8 (soft tissue window by default)."""
    lo = window_center - window_width // 2
    hi = window_center + window_width // 2
    vol = np.clip(volume, lo, hi)
    vol = ((vol - lo) / (hi - lo) * 255).astype(np.uint8)
    return vol


def slice_to_rgb(sl: np.ndarray) -> np.ndarray:
    """2D uint8 slice → 3-channel RGB as expected by SAM2."""
    return np.stack([sl, sl, sl], axis=-1)


def bbox_from_prior(prior: tuple, H: int, W: int) -> list[int]:
    """Convert fractional prior to pixel coords [x1, y1, x2, y2]."""
    x1 = int(prior[0] * W)
    y1 = int(prior[1] * H)
    x2 = int(prior[2] * W)
    y2 = int(prior[3] * H)
    return [x1, y1, x2, y2]


def predict_volume(predictor, volume_uint8: np.ndarray, class_id: int) -> np.ndarray:
    """Predict binary mask for one class across all axial slices."""
    H, W, D = volume_uint8.shape
    prior = CARDIAC_BBOX_PRIORS[class_id]
    bbox = np.array(bbox_from_prior(prior, H, W))
    mask_3d = np.zeros((H, W, D), dtype=np.uint8)

    for z in range(D):
        sl = slice_to_rgb(volume_uint8[:, :, z])
        predictor.set_image(sl)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],   # shape (1, 4)
            multimask_output=False,
        )
        mask_3d[:, :, z] = (masks[0] > 0).astype(np.uint8)

    return mask_3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/medsam2/zeroshot"))
    parser.add_argument("--medsam2-dir", type=Path, default=Path("models/medsam/MedSAM2"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt"))
    parser.add_argument("--config", default="configs/sam2.1_hiera_t512")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-center", type=int, default=40)
    parser.add_argument("--window-width", type=int, default=400)
    parser.add_argument("--cases", nargs="*", help="Specific case IDs to run (default: all)")
    args = parser.parse_args()

    # Add MedSAM2 to path
    sys.path.insert(0, str(args.medsam2_dir))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return

    model = build_sam2(args.config, str(args.checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.input_dir.glob("*_0000.nii.gz"))
    if not images:
        images = sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"MedSAM2 zero-shot: {len(images)} cases, device={args.device}")

    for img_path in images:
        case_id = img_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        out_path = args.output_dir / f"{case_id}.nii.gz"
        if out_path.exists():
            print(f"  [skip] {case_id}")
            continue

        print(f"  {case_id}...", end=" ", flush=True)
        try:
            nii = nib.load(str(img_path))
            volume = np.asarray(nii.dataobj, dtype=np.float32)
            vol_uint8 = normalize_to_uint8(volume, args.window_center, args.window_width)

            merged = np.zeros(volume.shape[:3], dtype=np.uint8)
            with torch.inference_mode():
                for class_id in LABEL_NAMES:
                    class_mask = predict_volume(predictor, vol_uint8, class_id)
                    merged[class_mask > 0] = class_id

            nib.save(nib.Nifti1Image(merged, nii.affine, nii.header), str(out_path))
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
