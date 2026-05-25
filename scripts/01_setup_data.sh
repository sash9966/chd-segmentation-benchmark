#!/usr/bin/env bash
# 01_setup_data.sh — Symlink nnUNet dataset + build benchmark splits
#
# Usage:
#   bash scripts/01_setup_data.sh \
#       /path/to/nnUNet_raw/Dataset001_CHD \
#       /path/to/nnUNet_preprocessed/Dataset001_CHD \
#       /path/to/splits_final.json

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DATASET="${1:?Usage: $0 <raw_dataset_dir> <preprocessed_dir> <splits_final_json>}"
PREPROCESSED="${2:?}"
SPLITS_FINAL="${3:?}"

echo "=== Step 1: Symlink data ==="
bash "$REPO/data/setup_symlinks.sh" "$RAW_DATASET" "$PREPROCESSED"

echo ""
echo "=== Step 2: Build benchmark splits ==="
python "$REPO/chd_benchmark/splits/build_splits.py" \
    --splits-final "$SPLITS_FINAL" \
    --images-ts-dir "$REPO/data/imagesTs" \
    --output-dir "$REPO/chd_benchmark/splits"

echo ""
echo "=== Step 3: QA check on training data ==="
python "$REPO/chd_benchmark/data_qa.py" \
    --images-dir "$REPO/data/imagesTr" \
    --labels-dir "$REPO/data/labelsTr" \
    --output-csv "$REPO/chd_benchmark/qa_report.csv"

echo ""
echo "Setup complete. Check chd_benchmark/qa_report.csv for any flagged cases."
