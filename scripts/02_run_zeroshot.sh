#!/usr/bin/env bash
# 02_run_zeroshot.sh — Run all models in zero-shot mode on the test set
#
# Each model uses its own conda environment. Activate before running:
#   conda activate <env>  && bash scripts/02_run_zeroshot.sh totalsegmentator
#   conda activate medsam && bash scripts/02_run_zeroshot.sh medsam
#   conda activate autoseg&& bash scripts/02_run_zeroshot.sh autoseg
#
# Or run all (requires you to switch envs):
#   bash scripts/02_run_zeroshot.sh all

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-all}"

run_totalseg() {
    echo "=== TotalSegmentator zero-shot ==="
    python "$REPO/models/totalsegmentator/run_zeroshot.py" \
        --input-dir "$REPO/data/imagesTs" \
        --output-dir "$REPO/results/totalsegmentator/zeroshot"
}

run_medsam() {
    echo "=== MedSAM zero-shot ==="
    python "$REPO/models/medsam/run_zeroshot.py" \
        --input-dir "$REPO/data/imagesTs" \
        --output-dir "$REPO/results/medsam/zeroshot" \
        --checkpoint "$REPO/models/medsam/checkpoints/medsam_vit_b.pth"
}

run_autoseg() {
    echo "=== Auto3DSeg zero-shot ==="
    python "$REPO/models/autoseg/run_zeroshot.py" \
        --input-dir "$REPO/data/imagesTs" \
        --output-dir "$REPO/results/autoseg/zeroshot" \
        --checkpoint "$REPO/models/autoseg/checkpoints/pretrained.pt"
}

case "$MODEL" in
    totalsegmentator) run_totalseg ;;
    medsam)           run_medsam ;;
    autoseg)          run_autoseg ;;
    all)
        run_totalseg
        run_medsam
        run_autoseg
        ;;
    *) echo "Unknown model: $MODEL. Choose: totalsegmentator | medsam | autoseg | all" && exit 1 ;;
esac

echo ""
echo "=== Evaluate zero-shot results ==="
bash "$REPO/scripts/04_evaluate.sh" zeroshot
