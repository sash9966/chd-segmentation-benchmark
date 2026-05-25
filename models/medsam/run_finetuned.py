#!/usr/bin/env python3
"""
MedSAM2 fine-tuned inference — merges per-class checkpoints into one label map.

Usage:
    python models/medsam/run_finetuned.py \
        --input-dir       data/imagesTs \
        --output-dir      results/medsam2/finetuned \
        --medsam2-dir     models/medsam/MedSAM2 \
        --checkpoint-dir  models/medsam/checkpoints/finetuned \
        --base-checkpoint models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt

Environment: medsam2_env
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from models.medsam.run_zeroshot import normalize_to_uint8, CARDIAC_BBOX_PRIORS

LABEL_IDS = list(range(1, 8))


def load_model(medsam2_dir: Path, base_ckpt: Path, finetuned_ckpt: Path, config: str, device: str):
    sys.path.insert(0, str(medsam2_dir))
    from sam2.build_sam import build_sam2
    model = build_sam2(config, str(base_ckpt), device=device)
    state = torch.load(str(finetuned_ckpt), map_location=device)
    model.load_state_dict(state)
    return model.eval()


def predict_volume(models: dict, volume_uint8: np.ndarray, device: str) -> np.ndarray:
    H, W, D = volume_uint8.shape
    pred = np.zeros((H, W, D), dtype=np.uint8)

    for class_id, model in models.items():
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(model)
        prior = CARDIAC_BBOX_PRIORS[class_id]
        bbox = np.array([
            int(prior[0] * W), int(prior[1] * H),
            int(prior[2] * W), int(prior[3] * H),
        ])
        class_mask = np.zeros((H, W, D), dtype=np.uint8)

        with torch.inference_mode():
            for z in range(D):
                sl = volume_uint8[:, :, z]
                rgb = np.stack([sl, sl, sl], axis=-1)
                predictor.set_image(rgb)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox[None, :],
                    multimask_output=False,
                )
                class_mask[:, :, z] = (masks[0] > 0).astype(np.uint8)

        pred[class_mask > 0] = class_id

    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",       type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir",      type=Path, default=Path("results/medsam2/finetuned"))
    parser.add_argument("--medsam2-dir",     type=Path, default=Path("models/medsam/MedSAM2"))
    parser.add_argument("--checkpoint-dir",  type=Path, default=Path("models/medsam/checkpoints/finetuned"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt"))
    parser.add_argument("--config",          default="configs/sam2.1_hiera_t512")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases",           nargs="*")
    args = parser.parse_args()

    sys.path.insert(0, str(args.medsam2_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for class_id in LABEL_IDS:
        ckpt = args.checkpoint_dir / f"medsam2_chd_class{class_id}.pt"
        if ckpt.exists():
            models[class_id] = load_model(
                args.medsam2_dir, args.base_checkpoint, ckpt, args.config, args.device
            )
        else:
            print(f"WARNING: no checkpoint for class {class_id} at {ckpt}")

    images = sorted(args.input_dir.glob("*_0000.nii.gz")) or sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"MedSAM2 fine-tuned inference: {len(images)} cases, {len(models)} class models loaded")

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
            vol_uint8 = normalize_to_uint8(volume)
            pred = predict_volume(models, vol_uint8, args.device)
            nib.save(nib.Nifti1Image(pred, nii.affine, nii.header), str(out_path))
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
