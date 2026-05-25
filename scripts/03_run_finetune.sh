#!/usr/bin/env bash
# 03_run_finetune.sh — Fine-tune all models and run inference on the test set
#
# Usage (activate the right env first):
#   conda activate totalseg && bash scripts/03_run_finetune.sh totalsegmentator
#   conda activate medsam   && bash scripts/03_run_finetune.sh medsam
#   conda activate autoseg  && bash scripts/03_run_finetune.sh autoseg
#
# Arguments:
#   $1: model name (totalsegmentator | medsam | autoseg)
#   $2: dataset ID for nnUNet-based models (default: 1)
#   $3: fold (default: 0)

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: $0 <model> [dataset_id] [fold]}"
DATASET_ID="${2:-1}"
FOLD="${3:-0}"

case "$MODEL" in
    totalsegmentator)
        echo "=== TotalSegmentator fine-tune + inference ==="
        conda activate totalseg
        python "$REPO/models/totalsegmentator/fine_tune.py" \
            --dataset-id "$DATASET_ID" \
            --fold "$FOLD" \
            --input-dir "$REPO/data/imagesTs" \
            --output-dir "$REPO/results/totalsegmentator/finetuned"
        ;;
    medsam)
        echo "=== MedSAM fine-tune (all classes) ==="
        conda activate medsam
        for class_id in 1 2 3 4 5 6 7; do
            echo "  Training class $class_id..."
            python "$REPO/models/medsam/fine_tune.py" \
                --splits-json "$REPO/chd_benchmark/splits/holdout.json" \
                --images-dir "$REPO/data/imagesTr" \
                --labels-dir "$REPO/data/labelsTr" \
                --checkpoint "$REPO/models/medsam/checkpoints/medsam_vit_b.pth" \
                --output-dir "$REPO/models/medsam/checkpoints/finetuned" \
                --class-id "$class_id"
        done
        echo "=== MedSAM fine-tuned inference ==="
        python "$REPO/models/medsam/run_finetuned.py" \
            --input-dir "$REPO/data/imagesTs" \
            --output-dir "$REPO/results/medsam/finetuned" \
            --checkpoint-dir "$REPO/models/medsam/checkpoints/finetuned" \
            --base-checkpoint "$REPO/models/medsam/checkpoints/medsam_vit_b.pth"
        ;;
    autoseg)
        echo "=== Auto3DSeg train + inference ==="
        conda activate autoseg
        python "$REPO/models/autoseg/train.py" \
            --splits-json "$REPO/chd_benchmark/splits/holdout.json" \
            --images-dir "$REPO/data/imagesTr" \
            --labels-dir "$REPO/data/labelsTr" \
            --output-dir "$REPO/models/autoseg/auto3dseg_work"
        # After training, Auto3DSeg writes predictions via its own runner.
        # Copy final ensemble predictions to results/:
        mkdir -p "$REPO/results/autoseg/finetuned"
        echo "NOTE: Copy Auto3DSeg predictions from models/autoseg/auto3dseg_work/ensemble_output/ to results/autoseg/finetuned/"
        ;;
    *) echo "Unknown model: $MODEL" && exit 1 ;;
esac

echo ""
echo "=== Evaluate fine-tuned results ==="
bash "$REPO/scripts/04_evaluate.sh" finetuned "$MODEL"
