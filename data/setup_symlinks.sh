#!/usr/bin/env bash
# setup_symlinks.sh — create symlinks from data/ to your nnUNet dataset
# so every model can read from data/ without duplicating files.
#
# Usage:
#   bash data/setup_symlinks.sh /path/to/nnUNet_raw/Dataset001_CHD [/path/to/nnUNet_preprocessed/Dataset001_CHD]
#
# After this runs, data/ will contain:
#   imagesTr/ labelsTr/ imagesTs/ labelsTs/ dataset.json  (symlinked)
#   nnUNet_preprocessed/  (symlinked if provided)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$REPO_ROOT/data"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/nnUNet_raw/DatasetXXX [/path/to/nnUNet_preprocessed/DatasetXXX]"
    exit 1
fi

RAW_DATASET="$1"
PREPROCESSED="${2:-}"

if [[ ! -d "$RAW_DATASET" ]]; then
    echo "ERROR: raw dataset directory not found: $RAW_DATASET"
    exit 1
fi

echo "=== Symlinking nnUNet raw dataset ==="
for subdir in imagesTr labelsTr imagesTs labelsTs; do
    src="$RAW_DATASET/$subdir"
    dst="$DATA_DIR/$subdir"
    if [[ -d "$src" ]]; then
        if [[ -L "$dst" ]]; then
            echo "  already linked: $subdir"
        else
            ln -s "$src" "$dst"
            echo "  linked: $subdir -> $src"
        fi
    else
        echo "  skipping (not found): $subdir"
    fi
done

# dataset.json
if [[ -f "$RAW_DATASET/dataset.json" ]]; then
    dst="$DATA_DIR/dataset.json"
    [[ -L "$dst" ]] || ln -s "$RAW_DATASET/dataset.json" "$dst"
    echo "  linked: dataset.json"
fi

# preprocessed (optional)
if [[ -n "$PREPROCESSED" ]]; then
    if [[ -d "$PREPROCESSED" ]]; then
        echo "=== Symlinking nnUNet preprocessed ==="
        dst="$DATA_DIR/nnUNet_preprocessed"
        if [[ -L "$dst" ]]; then
            echo "  already linked: nnUNet_preprocessed"
        else
            ln -s "$PREPROCESSED" "$dst"
            echo "  linked: nnUNet_preprocessed -> $PREPROCESSED"
        fi
    else
        echo "WARNING: preprocessed directory not found: $PREPROCESSED"
    fi
fi

echo ""
echo "Done. Contents of data/:"
ls -la "$DATA_DIR/"
