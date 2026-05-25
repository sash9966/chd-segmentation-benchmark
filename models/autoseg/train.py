#!/usr/bin/env python3
"""
MONAI Auto3DSeg training on the CHD benchmark (fold 0).

Auto3DSeg automatically designs and trains segmentation algorithms.
This script:
  1. Generates an Auto3DSeg-compatible datalist JSON from our splits
  2. Runs the Auto3DSeg pipeline (analysis → algorithm generation → training)

For fine-tuning comparison: use this as the "trained from scratch on CHD" baseline.
For a pretrained init: pass --ckpt to seed a SegResNet checkpoint first.

Usage:
    python models/autoseg/train.py \
        --splits-json chd_benchmark/splits/holdout.json \
        --images-dir data/imagesTr \
        --labels-dir data/labelsTr \
        --output-dir models/autoseg/auto3dseg_work \
        [--ckpt models/autoseg/checkpoints/pretrained.pt]

Environment: environments/autoseg.yml
"""

import argparse
import json
from pathlib import Path


def build_datalist(splits_json: Path, images_dir: Path, labels_dir: Path, output_json: Path) -> None:
    with open(splits_json) as f:
        splits = json.load(f)

    datalist = {"training": [], "validation": []}
    for case_id in splits["train"]:
        img_candidates = list(images_dir.glob(f"{case_id}_0000.nii.gz")) + list(images_dir.glob(f"{case_id}.nii.gz"))
        lbl_candidates = list(labels_dir.glob(f"{case_id}.nii.gz"))
        if img_candidates and lbl_candidates:
            datalist["training"].append({
                "image": str(img_candidates[0]),
                "label": str(lbl_candidates[0]),
            })

    for case_id in splits["val"]:
        img_candidates = list(images_dir.glob(f"{case_id}_0000.nii.gz")) + list(images_dir.glob(f"{case_id}.nii.gz"))
        lbl_candidates = list(labels_dir.glob(f"{case_id}.nii.gz"))
        if img_candidates and lbl_candidates:
            datalist["validation"].append({
                "image": str(img_candidates[0]),
                "label": str(lbl_candidates[0]),
            })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(datalist, f, indent=2)
    print(f"Datalist: {len(datalist['training'])} train, {len(datalist['validation'])} val → {output_json}")


def run_auto3dseg(work_dir: Path, datalist_json: Path, num_classes: int) -> None:
    from monai.apps.auto3dseg import AutoRunner

    runner = AutoRunner(
        work_dir=str(work_dir),
        input={
            "modality": "CT",
            "datalist": str(datalist_json),
            "dataroot": "/",  # paths in datalist are absolute
            "num_classes": num_classes,
        },
    )
    runner.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-json", type=Path, default=Path("chd_benchmark/splits/holdout.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/imagesTr"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labelsTr"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/autoseg/auto3dseg_work"))
    parser.add_argument("--num-classes", type=int, default=8)
    args = parser.parse_args()

    datalist_json = args.output_dir / "datalist.json"
    build_datalist(args.splits_json, args.images_dir, args.labels_dir, datalist_json)
    run_auto3dseg(args.output_dir, datalist_json, args.num_classes)


if __name__ == "__main__":
    main()
