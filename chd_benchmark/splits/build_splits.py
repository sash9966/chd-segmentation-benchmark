#!/usr/bin/env python3
"""
build_splits.py — Convert nnUNetv2 split files to benchmark-standard JSON format.

All models in the benchmark must use these splits for fair comparison.
Do NOT regenerate splits; always convert from the existing nnUNetv2 files.

Usage:
    python splits/build_splits.py \
        --splits-final /path/to/splits_final.json \
        --images-ts-dir /path/to/imagesTs \
        --output-dir splits/

Outputs:
    splits/folds.json    — all CV folds from splits_final.json
    splits/holdout.json  — single train/val/test using fold 0 + imagesTs test set
"""

import argparse
import json
import re
import sys
from pathlib import Path


def strip_modality_suffix(name: str) -> str:
    """
    imageCHD_0001_0000.nii.gz  →  imageCHD_0001
    imageCHD_0001.nii.gz       →  imageCHD_0001
    imageCHD_0001              →  imageCHD_0001   (already clean)
    """
    name = re.sub(r"_\d{4}\.nii\.gz$", "", name)  # strip _0000.nii.gz modality suffix
    name = re.sub(r"\.nii\.gz$", "", name)          # strip bare .nii.gz
    return name


def clean_case_list(cases: list[str]) -> list[str]:
    return [strip_modality_suffix(c) for c in cases]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert nnUNetv2 splits_final.json + imagesTs to benchmark JSON splits"
    )
    parser.add_argument(
        "--splits-final",
        type=Path,
        required=True,
        help="Path to nnUNetv2 splits_final.json",
    )
    parser.add_argument(
        "--images-ts-dir",
        type=Path,
        required=True,
        help="Path to imagesTs/ directory containing test .nii.gz files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Directory to write folds.json and holdout.json (default: splits/)",
    )
    args = parser.parse_args()

    # --- Load splits_final.json ---
    if not args.splits_final.exists():
        print(f"ERROR: splits_final.json not found at {args.splits_final}", file=sys.stderr)
        sys.exit(1)

    with open(args.splits_final) as f:
        raw_splits = json.load(f)

    # nnUNetv2 format: list of dicts with "train" and "val" keys
    if not isinstance(raw_splits, list) or not raw_splits:
        print("ERROR: splits_final.json must be a non-empty list of fold dicts", file=sys.stderr)
        sys.exit(1)

    # --- Load test cases from imagesTs ---
    if not args.images_ts_dir.exists():
        print(f"ERROR: imagesTs directory not found at {args.images_ts_dir}", file=sys.stderr)
        sys.exit(1)

    test_files = sorted(args.images_ts_dir.glob("*.nii.gz"))
    if not test_files:
        print(f"WARNING: no .nii.gz files found in {args.images_ts_dir}", file=sys.stderr)
    test_cases = [strip_modality_suffix(p.name) for p in test_files]

    # --- Build folds.json ---
    folds = {}
    for i, fold in enumerate(raw_splits):
        if "train" not in fold or "val" not in fold:
            print(f"WARNING: fold {i} missing 'train' or 'val' key — skipping", file=sys.stderr)
            continue
        folds[f"fold_{i}"] = {
            "train": clean_case_list(fold["train"]),
            "val": clean_case_list(fold["val"]),
        }

    # --- Build holdout.json (fold 0 train/val + imagesTs test) ---
    fold0 = folds.get("fold_0")
    if fold0 is None:
        print("ERROR: fold_0 not found after parsing splits_final.json", file=sys.stderr)
        sys.exit(1)

    holdout = {
        "train": fold0["train"],
        "val": fold0["val"],
        "test": test_cases,
    }

    # --- Write outputs ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = args.output_dir / "folds.json"
    holdout_path = args.output_dir / "holdout.json"

    with open(folds_path, "w") as f:
        json.dump(folds, f, indent=2)

    with open(holdout_path, "w") as f:
        json.dump(holdout, f, indent=2)

    # --- Summary ---
    print("=== Split Summary ===")
    for fold_name, fold_data in folds.items():
        print(f"  {fold_name}: {len(fold_data['train'])} train  |  {len(fold_data['val'])} val")
    print(f"  holdout: {len(holdout['train'])} train  |  {len(holdout['val'])} val  |  {len(holdout['test'])} test")

    all_cases = set()
    for fold_data in folds.values():
        all_cases.update(fold_data["train"])
        all_cases.update(fold_data["val"])
    all_cases.update(holdout["test"])
    print(f"  Total unique cases across all splits: {len(all_cases)}")

    print(f"\nWrote: {folds_path}")
    print(f"Wrote: {holdout_path}")


if __name__ == "__main__":
    main()
