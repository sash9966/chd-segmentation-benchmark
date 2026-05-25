#!/usr/bin/env bash
# 04_evaluate.sh — Compute metrics for all available predictions, then summarize
#
# Usage:
#   bash scripts/04_evaluate.sh [zeroshot|finetuned|all]  [model]
#
# Examples:
#   bash scripts/04_evaluate.sh all                        # all models, both variants
#   bash scripts/04_evaluate.sh zeroshot totalsegmentator  # one model, zero-shot only
#   bash scripts/04_evaluate.sh all                        # run summarize at the end

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-all}"
MODEL_FILTER="${2:-}"

MODELS=(totalsegmentator medsam autoseg)
VARIANTS=(zeroshot finetuned)

[[ "$VARIANT" != "all" ]] && VARIANTS=("$VARIANT")
[[ -n "$MODEL_FILTER" ]] && MODELS=("$MODEL_FILTER")

for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        pred_dir="$REPO/results/$model/$variant"
        csv_out="$REPO/results/$model/$variant/metrics.csv"

        [[ -d "$pred_dir" ]] || continue
        # Skip if no predictions exist
        n_preds=$(find "$pred_dir" -name "*.nii.gz" 2>/dev/null | wc -l)
        [[ "$n_preds" -eq 0 ]] && continue

        echo "=== $model / $variant ($n_preds predictions) ==="
        python "$REPO/evaluation/compute_metrics.py" \
            --pred-dir "$pred_dir" \
            --gt-dir "$REPO/data/labelsTs" \
            --output-csv "$csv_out" \
            --splits-json "$REPO/chd_benchmark/splits/holdout.json"
        echo ""
    done
done

echo "=== Comparison Summary ==="
python "$REPO/evaluation/summarize.py" \
    --results-dir "$REPO/results" \
    --output-csv "$REPO/results/comparison_table.csv"
