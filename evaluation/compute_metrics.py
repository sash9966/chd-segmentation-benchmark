#!/usr/bin/env python3
"""
compute_metrics.py — Dice and HD95 per class for one model's predictions.

Usage:
    python evaluation/compute_metrics.py \
        --pred-dir results/totalsegmentator/zeroshot \
        --gt-dir data/labelsTs \
        --output-csv results/totalsegmentator/zeroshot/metrics.csv \
        [--splits-json chd_benchmark/splits/holdout.json]  # optional: filter to test cases

Outputs one CSV row per case with Dice and HD95 for each of the 7 CHD classes.
"""

import argparse
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.ndimage.morphology import distance_transform_edt

warnings.filterwarnings("ignore", category=FutureWarning)

LABEL_NAMES = {
    1: "left_ventricle",
    2: "right_ventricle",
    3: "left_atrium",
    4: "right_atrium",
    5: "myocardium",
    6: "aorta",
    7: "pulmonary_artery",
}


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2 * inter / denom if denom > 0 else float("nan")


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> float:
    """
    Symmetric 95th-percentile Hausdorff distance in mm.
    spacing: (sx, sy, sz) voxel sizes in mm.
    """
    if not pred.any() or not gt.any():
        return float("nan")

    def surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Distance from surface of a to nearest point of b
        dt_b = distance_transform_edt(~b, sampling=spacing)
        surface_a = a ^ (pred & True)  # reuse variable; compute border
        # Use morphological erosion to get surface voxels
        from scipy.ndimage import binary_erosion
        interior_a = binary_erosion(a)
        border_a = a & ~interior_a
        return dt_b[border_a]

    d_pred_to_gt = surface_distances(pred, gt)
    d_gt_to_pred = surface_distances(gt, pred)

    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return float("nan")

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_case_metrics(pred_path: Path, gt_path: Path) -> dict:
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))

    pred = np.asarray(pred_nii.dataobj, dtype=np.uint8)
    gt = np.asarray(gt_nii.dataobj, dtype=np.uint8)

    # Resample pred to gt space if shapes differ
    if pred.shape != gt.shape:
        from scipy.ndimage import zoom
        factors = [g / p for g, p in zip(gt.shape, pred.shape)]
        pred = zoom(pred, factors, order=0).astype(np.uint8)

    spacing = tuple(float(v) for v in gt_nii.header.get_zooms()[:3])

    row: dict = {}
    for cls_id, cls_name in LABEL_NAMES.items():
        p = pred == cls_id
        g = gt == cls_id
        row[f"{cls_name}_dice"] = dice(p, g)
        row[f"{cls_name}_hd95"] = hausdorff_95(p, g, spacing)

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, default=None,
                        help="If provided, only evaluate cases listed under 'test'")
    args = parser.parse_args()

    case_filter = None
    if args.splits_json and args.splits_json.exists():
        import json
        with open(args.splits_json) as f:
            splits = json.load(f)
        case_filter = set(splits.get("test", []))

    gt_files = sorted(args.gt_dir.glob("*.nii.gz"))
    if case_filter:
        gt_files = [p for p in gt_files if p.name.replace(".nii.gz", "") in case_filter]

    rows = []
    for gt_path in gt_files:
        case_id = gt_path.name.replace(".nii.gz", "")
        pred_path = args.pred_dir / gt_path.name
        if not pred_path.exists():
            print(f"  [missing prediction] {case_id}")
            continue
        print(f"  {case_id}...", end=" ", flush=True)
        try:
            metrics = compute_case_metrics(pred_path, gt_path)
            metrics["case_id"] = case_id
            rows.append(metrics)
            mean_dice = np.nanmean([v for k, v in metrics.items() if k.endswith("_dice")])
            print(f"mean Dice={mean_dice:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        # Move case_id to first column
        cols = ["case_id"] + [c for c in df.columns if c != "case_id"]
        df = df[cols]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\n=== Metric Summary ===")
    for cls_name in LABEL_NAMES.values():
        d = df.get(f"{cls_name}_dice", pd.Series(dtype=float))
        h = df.get(f"{cls_name}_hd95", pd.Series(dtype=float))
        print(f"  {cls_name:<22} Dice={d.mean():.3f}±{d.std():.3f}  HD95={h.mean():.2f}mm")
    print(f"\nReport written to {args.output_csv}")


if __name__ == "__main__":
    main()
