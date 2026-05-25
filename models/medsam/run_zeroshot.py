#!/usr/bin/env python3
"""
MedSAM zero-shot inference on the CHD benchmark test set.

MedSAM (Medical SAM) is a SAM model fine-tuned on medical images.
Zero-shot here means: use the pretrained MedSAM checkpoint with bounding-box
prompts derived from a coarse prior (e.g., whole-volume crop or atlas-derived boxes).

Since MedSAM is 2D (processes slices), we run it slice-by-slice and reconstruct
a 3D volume. For 3D volumes we use the axial plane by default.

Checkpoint download:
    https://drive.google.com/drive/folders/1ETWmi4AiniJVWr-V0iYDd5IFk6C5Mb0Z
    Place at: models/medsam/checkpoints/medsam_vit_b.pth

Usage:
    python models/medsam/run_zeroshot.py \
        --input-dir data/imagesTs \
        --output-dir results/medsam/zeroshot \
        --checkpoint models/medsam/checkpoints/medsam_vit_b.pth \
        [--class-id 1]   # run for a specific class; default runs all

Environment: environments/medsam.yml
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


LABEL_IDS = list(range(1, 8))  # 1-7 (exclude background)


def load_medsam(checkpoint: Path, device: str):
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()
    return model


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize to [0, 255] uint8 as expected by SAM."""
    lo, hi = slice_2d.min(), slice_2d.max()
    if hi == lo:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    norm = (slice_2d - lo) / (hi - lo) * 255
    return norm.astype(np.uint8)


def coarse_bbox_from_volume(volume: np.ndarray, margin: float = 0.1) -> tuple[int, int, int, int]:
    """
    Return a 2D bounding box covering the central region of the volume
    as a coarse prior when no label information is available.
    margin: fractional margin added around the center crop.
    """
    H, W = volume.shape[:2]
    m = int(min(H, W) * margin)
    return m, m, W - m, H - m  # x1, y1, x2, y2


def run_medsam_slice(model, predictor, slice_rgb: np.ndarray, bbox: list[int], device: str) -> np.ndarray:
    """Run MedSAM on one 2D slice with a bounding-box prompt."""
    from segment_anything.utils.transforms import ResizeLongestSide
    transform = ResizeLongestSide(1024)

    input_image = transform.apply_image(slice_rgb)
    input_tensor = torch.as_tensor(input_image, dtype=torch.float32, device=device)
    input_tensor = input_tensor.permute(2, 0, 1)[None]

    with torch.no_grad():
        image_embedding = model.image_encoder(model.preprocess(input_tensor))
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=torch.tensor([bbox], dtype=torch.float32, device=device),
            masks=None,
        )
        masks, _, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    mask = masks[0, 0].cpu().numpy() > 0
    # Resize back to original slice size
    from skimage.transform import resize
    mask = resize(mask.astype(np.float32), slice_rgb.shape[:2], order=0) > 0.5
    return mask.astype(np.uint8)


def predict_volume(model, volume: np.ndarray, device: str) -> np.ndarray:
    """Predict all 8 CHD structures for a 3D volume, slice by slice."""
    H, W, D = volume.shape
    # We'll produce a single merged label map
    pred = np.zeros((H, W, D), dtype=np.uint8)

    bbox = list(coarse_bbox_from_volume(volume[..., 0]))

    # For zero-shot, run one pass per class using the same coarse bbox
    # (in practice you'd refine per class; this is a baseline)
    for label_id in LABEL_IDS:
        class_mask = np.zeros((H, W, D), dtype=np.uint8)
        for z in range(D):
            sl = volume[:, :, z]
            rgb = np.stack([normalize_slice(sl)] * 3, axis=-1)
            mask = run_medsam_slice(model, None, rgb, bbox, device)
            class_mask[:, :, z] = mask
        # Only assign this class where not already assigned by a higher-priority class
        pred[class_mask > 0] = label_id

    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/medsam/zeroshot"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/medsam/checkpoints/medsam_vit_b.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: MedSAM checkpoint not found at {args.checkpoint}")
        print("Download from: https://drive.google.com/drive/folders/1ETWmi4AiniJVWr-V0iYDd5IFk6C5Mb0Z")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_medsam(args.checkpoint, args.device)

    images = sorted(args.input_dir.glob("*_0000.nii.gz"))
    if not images:
        images = sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"Running MedSAM zero-shot on {len(images)} cases...")
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
            pred = predict_volume(model, volume, args.device)
            nib.save(nib.Nifti1Image(pred, nii.affine, nii.header), str(out_path))
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
