#!/usr/bin/env python3
"""
MedSAM fine-tuned inference — merges per-class checkpoints into one label map.

Usage:
    python models/medsam/run_finetuned.py \
        --input-dir data/imagesTs \
        --output-dir results/medsam/finetuned \
        --checkpoint-dir models/medsam/checkpoints/finetuned \
        --base-checkpoint models/medsam/checkpoints/medsam_vit_b.pth

Environment: environments/medsam.yml
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

LABEL_IDS = list(range(1, 8))


def load_model(base_checkpoint: Path, finetuned_checkpoint: Path, device: str):
    from segment_anything import sam_model_registry
    model = sam_model_registry["vit_b"](checkpoint=str(base_checkpoint))
    state = torch.load(str(finetuned_checkpoint), map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def predict_volume(models: dict, volume: np.ndarray, device: str) -> np.ndarray:
    from skimage.transform import resize
    H, W, D = volume.shape
    pred = np.zeros((H, W, D), dtype=np.uint8)

    for class_id, model in models.items():
        class_mask = np.zeros((H, W, D), dtype=np.uint8)
        for z in range(D):
            sl = volume[:, :, z]
            lo, hi = sl.min(), sl.max()
            sl_norm = ((sl - lo) / (hi - lo + 1e-6) * 255).astype(np.float32)
            sl_r = resize(sl_norm, (1024, 1024), order=1, preserve_range=True)
            image_tensor = torch.from_numpy(np.stack([sl_r, sl_r, sl_r], axis=0)).float().unsqueeze(0).to(device)

            m = int(1024 * 0.1)
            bbox = torch.tensor([[m, m, 1024 - m, 1024 - m]], dtype=torch.float32, device=device)

            with torch.no_grad():
                emb = model.image_encoder(model.preprocess(image_tensor))
                sparse, dense = model.prompt_encoder(points=None, boxes=bbox.unsqueeze(1), masks=None)
                masks, _ = model.mask_decoder(
                    image_embeddings=emb,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )[:2]
            mask = (torch.sigmoid(masks[0, 0]) > 0.5).cpu().numpy()
            mask = resize(mask.astype(np.float32), (H, W), order=0) > 0.5
            class_mask[:, :, z] = mask.astype(np.uint8)
        pred[class_mask > 0] = class_id

    return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/medsam/finetuned"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/medsam/checkpoints/finetuned"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("models/medsam/checkpoints/medsam_vit_b.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for class_id in LABEL_IDS:
        ckpt = args.checkpoint_dir / f"medsam_chd_class{class_id}.pth"
        if ckpt.exists():
            models[class_id] = load_model(args.base_checkpoint, ckpt, args.device)
        else:
            print(f"WARNING: no checkpoint for class {class_id} at {ckpt}")

    images = sorted(args.input_dir.glob("*_0000.nii.gz")) or sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"Running fine-tuned MedSAM on {len(images)} cases...")
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
            pred = predict_volume(models, volume, args.device)
            nib.save(nib.Nifti1Image(pred, nii.affine, nii.header), str(out_path))
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
