#!/usr/bin/env python3
"""
summarize.py — Aggregate metrics CSVs from all models into a comparison table.

Usage:
    python evaluation/summarize.py \
        --results-dir results \
        --output-csv results/comparison_table.csv \
        [--latex]  # also print a LaTeX table

Expects results/<model>/<variant>/metrics.csv
e.g.:
    results/totalsegmentator/zeroshot/metrics.csv
    results/totalsegmentator/finetuned/metrics.csv
    results/medsam/zeroshot/metrics.csv
    results/medsam/finetuned/metrics.csv
    results/autoseg/zeroshot/metrics.csv
    results/autoseg/finetuned/metrics.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAMES = [
    "left_ventricle", "right_ventricle", "left_atrium", "right_atrium",
    "myocardium", "aorta", "pulmonary_artery",
]


def load_model_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    row: dict = {}
    for cls in LABEL_NAMES:
        dice_col = f"{cls}_dice"
        hd95_col = f"{cls}_hd95"
        if dice_col in df.columns:
            row[f"{cls}_dice"] = df[dice_col].mean()
        if hd95_col in df.columns:
            row[f"{cls}_hd95"] = df[hd95_col].mean()
    dice_vals = [row[f"{cls}_dice"] for cls in LABEL_NAMES if f"{cls}_dice" in row]
    hd95_vals = [row[f"{cls}_hd95"] for cls in LABEL_NAMES if f"{cls}_hd95" in row]
    row["mean_dice"] = float(np.nanmean(dice_vals)) if dice_vals else float("nan")
    row["mean_hd95"] = float(np.nanmean(hd95_vals)) if hd95_vals else float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/comparison_table.csv"))
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    records = []
    for csv_path in sorted(args.results_dir.glob("*/*/metrics.csv")):
        parts = csv_path.parts
        # results/<model>/<variant>/metrics.csv
        idx = parts.index(args.results_dir.name)
        model = parts[idx + 1]
        variant = parts[idx + 2]
        metrics = load_model_metrics(csv_path)
        records.append({"model": model, "variant": variant, **metrics})

    if not records:
        print(f"No metrics.csv files found under {args.results_dir}")
        return

    df = pd.DataFrame(records)
    df = df.sort_values(["model", "variant"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Console summary
    display_cols = ["model", "variant", "mean_dice", "mean_hd95"] + \
                   [f"{c}_dice" for c in LABEL_NAMES]
    display = df[[c for c in display_cols if c in df.columns]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n=== Comparison Table ===")
    print(display.to_string(index=False))

    if args.latex:
        print("\n=== LaTeX Table ===")
        latex_cols = ["model", "variant", "mean_dice", "mean_hd95"]
        print(display[latex_cols].to_latex(index=False, float_format="%.3f"))

    print(f"\nFull table saved to {args.output_csv}")


if __name__ == "__main__":
    main()
