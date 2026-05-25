#!/bin/bash
# =============================================================================
#  medsam2_finetune.sh
#  MedSAM2 — fine-tune on Dataset030_imageCHD_HU then inference on imagesTs
#
#  Steps
#    Phase 1: Verify checkpoint, data, and splits
#    Phase 2: Fine-tune mask decoder for each of 7 CHD classes
#             (image encoder frozen; one .pt checkpoint per class)
#    Phase 3: Fine-tuned inference on imagesTs (merge 7 class masks)
#    Phase 4: Compute Dice + HD95 metrics vs labelsTs
#
#  RESUME SUPPORT
#    Per-class training markers: p2_finetune_classN.done
#    Re-submitting skips any class already trained.
#    Fine-tuned inference is also skipped per case if .nii.gz exists.
# =============================================================================
#SBATCH --job-name=medsam2-ft
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/medsam2_finetune_%j.out
#SBATCH --error=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/medsam2_finetune_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/miniconda/envs/medsam2_env
hash -r

export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────
# 2.  Configuration
# ─────────────────────────────────────────────
DATASET_NAME="Dataset030_imageCHD_HU"
REPO="/scratch/users/sastocke/chd-segmentation-benchmark"
IN_DIR="${REPO}/data/imagesTs"
OUT_DIR="${REPO}/results/medsam2/finetuned"
METRICS_CSV="${OUT_DIR}/metrics.csv"
SPLITS_JSON="${REPO}/chd_benchmark/splits/holdout.json"
MEDSAM2_DIR="${REPO}/models/medsam/MedSAM2"
BASE_CKPT="${MEDSAM2_DIR}/checkpoints/MedSAM2_latest.pt"
FINETUNE_CKPT_DIR="${REPO}/models/medsam/checkpoints/finetuned"
CKPT_DIR="${REPO}/results/medsam2/.checkpoints/finetune"
CLASS_NAMES=(""  "left_ventricle" "right_ventricle" "left_atrium" "right_atrium" \
                 "myocardium" "aorta" "pulmonary_artery")
START_TS=$(date +%s)

mkdir -p "${CKPT_DIR}" "${OUT_DIR}" "${FINETUNE_CKPT_DIR}"

# ─────────────────────────────────────────────
# 3.  Resume helpers
# ─────────────────────────────────────────────
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

# ─────────────────────────────────────────────
# 4.  Banner
# ─────────────────────────────────────────────
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  MedSAM2  —  FINE-TUNE  —  START                                ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}"
    printf "║  %-66s ║\n" "Strategy   : Freeze image encoder, train mask decoder"
    printf "║  %-66s ║\n" "Classes    : 7 binary models (LV RV LA RA Myo AO PA)"
    printf "║  %-66s ║\n" "Splits     : holdout.json fold 0  (train/val from nnUNet)"
    printf "║  %-66s ║\n" "Base ckpt  : ${BASE_CKPT}"
    printf "║  %-66s ║\n" "FT ckpts   : ${FINETUNE_CKPT_DIR}/"
    printf "║  %-66s ║\n" "Input      : ${IN_DIR}"
    printf "║  %-66s ║\n" "Output     : ${OUT_DIR}"
    printf "║  %-66s ║\n" "Metrics    : ${METRICS_CSV}"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "  Completed steps (from previous runs, if any):"
    ls "${CKPT_DIR}/"*.done 2>/dev/null \
        | xargs -I{} basename {} .done \
        | sort | sed 's/^/    [DONE] /' \
        || echo "    (none — fresh run)"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_footer() {
    local elapsed=$(( $(date +%s) - START_TS ))
    local hh=$(( elapsed / 3600 ))
    local mm=$(( (elapsed % 3600) / 60 ))
    local ss=$(( elapsed % 60 ))
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  MedSAM2  —  FINE-TUNE  —  COMPLETE                             ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "FT ckpts   : ${FINETUNE_CKPT_DIR}/medsam2_chd_classN.pt"
    printf "║  %-66s ║\n" "Predictions: ${OUT_DIR}/"
    printf "║  %-66s ║\n" "Metrics    : ${METRICS_CSV}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner
cd "${REPO}"

# ─────────────────────────────────────────────
# Phase 1 — Verify checkpoint, data, splits
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Verify checkpoint, data, and splits"
echo "================================================================"
if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "ERROR: MedSAM2 checkpoint not found: ${BASE_CKPT}"
    exit 1
fi
if [[ ! -d "${REPO}/data/imagesTr" ]]; then
    echo "ERROR: imagesTr not found. Run: bash data/setup_symlinks.sh"
    exit 1
fi
if [[ ! -f "${SPLITS_JSON}" ]]; then
    echo "ERROR: splits not found at ${SPLITS_JSON}"
    exit 1
fi
N_TR=$(ls "${REPO}/data/imagesTr"/*.nii.gz 2>/dev/null | wc -l)
N_TS=$(ls "${REPO}/data/imagesTs"/*.nii.gz 2>/dev/null | wc -l)
echo "  Base ckpt  : $(du -sh ${BASE_CKPT} | cut -f1)  ${BASE_CKPT}"
echo "  Train cases: ${N_TR} | Test cases: ${N_TS}"
echo "[OK] Phase 1"

# ─────────────────────────────────────────────
# Phase 2 — Fine-tune mask decoder (7 classes)
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 2: Fine-tune MedSAM2 mask decoder — 7 classes"
echo "  Splits     : ${SPLITS_JSON}"
echo "  Output dir : ${FINETUNE_CKPT_DIR}"
echo "================================================================"
for CLASS_ID in 1 2 3 4 5 6 7; do
    KEY="p2_finetune_class${CLASS_ID}"
    LABEL="${CLASS_NAMES[$CLASS_ID]}"
    if is_done "${KEY}"; then
        echo "[SKIP] Class ${CLASS_ID} (${LABEL})"
    else
        echo ""
        echo "--- Class ${CLASS_ID}: ${LABEL} ---"
        python models/medsam/fine_tune.py \
            --splits-json   "${SPLITS_JSON}" \
            --images-dir    data/imagesTr \
            --labels-dir    data/labelsTr \
            --medsam2-dir   "${MEDSAM2_DIR}" \
            --checkpoint    "${BASE_CKPT}" \
            --output-dir    "${FINETUNE_CKPT_DIR}" \
            --class-id      ${CLASS_ID} \
            --device cuda
        mark_done "${KEY}"
        echo "[OK] Class ${CLASS_ID} (${LABEL})"
    fi
done
echo ""
echo "[OK] Phase 2 — all 7 classes trained"

# ─────────────────────────────────────────────
# Phase 3 — Fine-tuned inference
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 3: Fine-tuned inference on test set"
echo "  Checkpoint dir : ${FINETUNE_CKPT_DIR}"
echo "  Input          : ${IN_DIR}"
echo "  Output         : ${OUT_DIR}"
echo "================================================================"
if is_done "p3_inference"; then
    echo "[SKIP] Phase 3: inference already done"
else
    python models/medsam/run_finetuned.py \
        --input-dir       "${IN_DIR}" \
        --output-dir      "${OUT_DIR}" \
        --medsam2-dir     "${MEDSAM2_DIR}" \
        --checkpoint-dir  "${FINETUNE_CKPT_DIR}" \
        --base-checkpoint "${BASE_CKPT}" \
        --device cuda
    mark_done "p3_inference"
    echo "[OK] Phase 3"
fi

# ─────────────────────────────────────────────
# Phase 4 — Metrics
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 4: Compute Dice + HD95 metrics"
echo "  Predictions : ${OUT_DIR}"
echo "  Ground truth: ${REPO}/data/labelsTs"
echo "  Output CSV  : ${METRICS_CSV}"
echo "================================================================"
if is_done "p4_metrics"; then
    echo "[SKIP] Phase 4: metrics already computed"
else
    python evaluation/compute_metrics.py \
        --pred-dir    "${OUT_DIR}" \
        --gt-dir      data/labelsTs \
        --output-csv  "${METRICS_CSV}" \
        --splits-json "${SPLITS_JSON}"
    mark_done "p4_metrics"
    echo "[OK] Phase 4"
fi

print_footer
