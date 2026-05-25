#!/usr/bin/env python3
"""
Fine-tune TotalSegmentator on the CHD benchmark using nnUNet.

TotalSegmentator is built on top of nnUNet. Fine-tuning means:
  1. Import TS pretrained weights into an nnUNet experiment
  2. Run nnUNet training (continues from the imported checkpoint)
  3. Run inference with the fine-tuned model

This script wraps the nnUNet CLI calls. It assumes your dataset is already
registered as a nnUNet dataset and preprocessed.

Usage:
    python models/totalsegmentator/fine_tune.py \
        --dataset-id 001 \
        --fold 0 \
        --task heartchambers_highres \
        [--continue-training]   # resume a partially completed run

After training, run inference with:
    python models/totalsegmentator/fine_tune.py --infer-only \
        --dataset-id 001 --fold 0 \
        --input-dir data/imagesTs \
        --output-dir results/totalsegmentator/finetuned

Environment: environments/totalsegmentator.yml
"""

import argparse
import subprocess
from pathlib import Path


TRAINER = "nnUNetTrainer"
PLANNER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"


def import_ts_weights(dataset_id: int, task: str) -> None:
    """Use totalseg_import_weights to seed the nnUNet experiment from TS pretrained."""
    cmd = [
        "totalseg_import_weights",
        "--dataset-id", str(dataset_id),
        "--task", task,
    ]
    print("Importing TotalSegmentator weights:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train(dataset_id: int, fold: int, continue_training: bool) -> None:
    dataset_str = f"Dataset{dataset_id:03d}"
    cmd = [
        "nnUNetv2_train",
        dataset_str,
        CONFIGURATION,
        str(fold),
        "--trainer", TRAINER,
        "--plans-identifier", PLANNER,
    ]
    if continue_training:
        cmd.append("--continue-training")
    print("Training:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def predict(dataset_id: int, fold: int, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_str = f"Dataset{dataset_id:03d}"
    cmd = [
        "nnUNetv2_predict",
        "-d", dataset_str,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-f", str(fold),
        "-tr", TRAINER,
        "-p", PLANNER,
        "-c", CONFIGURATION,
    ]
    print("Predicting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True, help="nnUNet dataset ID (e.g. 1)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--task", default="heartchambers_highres", help="TS task to import weights from")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--skip-import", action="store_true", help="Skip weight import (if already done)")
    parser.add_argument("--infer-only", action="store_true", help="Skip training, only run prediction")
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/totalsegmentator/finetuned"))
    args = parser.parse_args()

    if not args.infer_only:
        if not args.skip_import:
            import_ts_weights(args.dataset_id, args.task)
        train(args.dataset_id, args.fold, args.continue_training)

    predict(args.dataset_id, args.fold, args.input_dir, args.output_dir)
    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
