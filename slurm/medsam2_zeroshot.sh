#!/bin/bash
# =============================================================================
#  medsam2_zeroshot.sh
#  MedSAM2 — zero-shot inference on Dataset030_imageCHD_HU
#
#  Steps
#    Phase 1: Verify checkpoint + test set
#    Phase 2: MedSAM2 zero-shot inference  (slice-by-slice, 7 classes)
#    Phase 3: Compute Dice + HD95 metrics vs labelsTs
#
#  RESUME SUPPORT
#    Each phase creates a .done marker. Re-submitting skips completed phases.
#    Individual case predictions are also skipped if the .nii.gz already exists.
# =============================================================================
#SBATCH --job-name=medsam2-zs
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/medsam2_zeroshot_%j.out
#SBATCH --error=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/medsam2_zeroshot_%j.err

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
OUT_DIR="${REPO}/results/medsam2/zeroshot"
METRICS_CSV="${OUT_DIR}/metrics.csv"
SPLITS_JSON="${REPO}/chd_benchmark/splits/holdout.json"
MEDSAM2_DIR="${REPO}/models/medsam/MedSAM2"
CHECKPOINT="${MEDSAM2_DIR}/checkpoints/MedSAM2_latest.pt"
CKPT_DIR="${REPO}/results/medsam2/.checkpoints/zeroshot"
START_TS=$(date +%s)

mkdir -p "${CKPT_DIR}" "${OUT_DIR}"

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
    echo "║  MedSAM2  —  ZERO-SHOT  —  START                                ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}"
    printf "║  %-66s ║\n" "Strategy   : SAM2ImagePredictor, axial slices, bbox priors"
    printf "║  %-66s ║\n" "Classes    : 7 (LV RV LA RA Myo AO PA)"
    printf "║  %-66s ║\n" "Checkpoint : ${CHECKPOINT}"
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
    echo "║  MedSAM2  —  ZERO-SHOT  —  COMPLETE                             ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
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
# Phase 1 — Verify checkpoint + test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Verify checkpoint and test set"
echo "================================================================"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: MedSAM2 checkpoint not found: ${CHECKPOINT}"
    echo "  wget -O ${CHECKPOINT} https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
    exit 1
fi
if [[ ! -d "${IN_DIR}" ]]; then
    echo "ERROR: imagesTs not found at ${IN_DIR}"
    exit 1
fi
N_CASES=$(ls "${IN_DIR}"/*.nii.gz 2>/dev/null | wc -l)
echo "  Checkpoint : $(du -sh ${CHECKPOINT} | cut -f1)  ${CHECKPOINT}"
echo "  Test cases : ${N_CASES}"
echo "[OK] Phase 1"

# ─────────────────────────────────────────────
# Phase 2 — Zero-shot inference
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 2: MedSAM2 zero-shot inference"
echo "  Input  : ${IN_DIR}"
echo "  Output : ${OUT_DIR}"
echo "  Device : CUDA"
echo "================================================================"
if is_done "p2_inference"; then
    echo "[SKIP] Phase 2: inference already done"
else
    python models/medsam/run_zeroshot.py \
        --input-dir   "${IN_DIR}" \
        --output-dir  "${OUT_DIR}" \
        --medsam2-dir "${MEDSAM2_DIR}" \
        --checkpoint  "${CHECKPOINT}" \
        --device cuda
    mark_done "p2_inference"
    echo "[OK] Phase 2"
fi

# ─────────────────────────────────────────────
# Phase 3 — Metrics
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 3: Compute Dice + HD95 metrics"
echo "  Predictions : ${OUT_DIR}"
echo "  Ground truth: ${REPO}/data/labelsTs"
echo "  Output CSV  : ${METRICS_CSV}"
echo "================================================================"
if is_done "p3_metrics"; then
    echo "[SKIP] Phase 3: metrics already computed"
else
    python evaluation/compute_metrics.py \
        --pred-dir    "${OUT_DIR}" \
        --gt-dir      data/labelsTs \
        --output-csv  "${METRICS_CSV}" \
        --splits-json "${SPLITS_JSON}"
    mark_done "p3_metrics"
    echo "[OK] Phase 3"
fi

print_footer
