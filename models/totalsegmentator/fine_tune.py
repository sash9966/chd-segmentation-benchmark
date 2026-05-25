#!/usr/bin/env python3
"""
Fine-tune / train nnUNet on Dataset030_imageCHD_HU.

This trains nnUNet from scratch on the CHD dataset using the same nnUNet
backbone that TotalSegmentator is built on. Together with the TotalSegmentator
zero-shot run, this gives a direct comparison:
  - Zero-shot: TotalSegmentator pretrained on general CT (heartchambers_highres)
  - Fine-tuned: nnUNet trained on CHD data (Dataset030_imageCHD_HU)

Dataset must already be preprocessed (splits_final.json exists in
nnUNet_preprocessed/Dataset030_imageCHD_HU/).

Usage:
    python models/totalsegmentator/fine_tune.py \
        --dataset-id 30 \
        --fold 0 \
        [--plans nnUNetResEncUNetMPlans]   # match your existing nnUNet setup
        [--trainer nnUNetTrainerDA5_200epochs]
        [--infer-only]
        [--input-dir data/imagesTs]
        [--output-dir results/totalsegmentator/finetuned]

Environment: totalseg_env  (has nnUNetv2 installed via TotalSegmentator)
"""

import argparse
import subprocess
from pathlib import Path


def train(dataset_id: int, fold: int, trainer: str, plans: str,
          config: str, continue_training: bool) -> None:
    dataset_str = f"Dataset{dataset_id:03d}"
    cmd = [
        "nnUNetv2_train",
        dataset_str,
        config,
        str(fold),
        "-tr", trainer,
        "-p", plans,
        "--npz",
    ]
    if continue_training:
        cmd.append("--c")
    print("Training:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def predict(dataset_id: int, fold: int, trainer: str, plans: str,
            config: str, input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_str = f"Dataset{dataset_id:03d}"
    cmd = [
        "nnUNetv2_predict",
        "-d", dataset_str,
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-f", str(fold),
        "-tr", trainer,
        "-p", plans,
        "-c", config,
    ]
    print("Predicting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id",        type=int, default=30)
    parser.add_argument("--fold",              type=int, default=0)
    parser.add_argument("--trainer",           default="nnUNetTrainer")
    parser.add_argument("--plans",             default="nnUNetPlans")
    parser.add_argument("--config",            default="3d_fullres")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--infer-only",        action="store_true")
    parser.add_argument("--input-dir",         type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir",        type=Path, default=Path("results/totalsegmentator/finetuned"))
    args = parser.parse_args()

    if not args.infer_only:
        train(args.dataset_id, args.fold, args.trainer, args.plans,
              args.config, args.continue_training)

    predict(args.dataset_id, args.fold, args.trainer, args.plans,
            args.config, args.input_dir, args.output_dir)
    print(f"\nPredictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
