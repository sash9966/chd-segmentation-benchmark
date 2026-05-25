#!/bin/bash
# =============================================================================
#  totalseg_zeroshot.sh
#  TotalSegmentator v2 — zero-shot inference on Dataset030_imageCHD_HU
#
#  Steps
#    Phase 1: Verify data symlinks + test set
#    Phase 2: TotalSegmentator heartchambers_highres inference (imagesTs)
#    Phase 3: Compute Dice + HD95 metrics vs labelsTs
#
#  RESUME SUPPORT
#    Each phase creates a .done marker in CKPT_DIR.
#    Re-submitting this script skips completed phases automatically.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs
# =============================================================================
#SBATCH --job-name=totalseg-zs
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/totalseg_zeroshot_%j.out
#SBATCH --error=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/totalseg_zeroshot_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/miniconda/envs/totalseg_env
hash -r

export TOTALSEG_WEIGHTS_PATH="/scratch/users/sastocke/chd-segmentation-benchmark/models/totalsegmentator/weights"
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────
# 2.  Configuration
# ─────────────────────────────────────────────
DATASET_NAME="Dataset030_imageCHD_HU"
REPO="/scratch/users/sastocke/chd-segmentation-benchmark"
IN_DIR="${REPO}/data/imagesTs"
OUT_DIR="${REPO}/results/totalsegmentator/zeroshot"
METRICS_CSV="${OUT_DIR}/metrics.csv"
SPLITS_JSON="${REPO}/chd_benchmark/splits/holdout.json"
CKPT_DIR="${REPO}/results/totalsegmentator/.checkpoints/zeroshot"
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
    echo "║  TotalSegmentator  —  ZERO-SHOT  —  START                       ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}"
    printf "║  %-66s ║\n" "Task       : heartchambers_highres"
    printf "║  %-66s ║\n" "Input      : ${IN_DIR}"
    printf "║  %-66s ║\n" "Output     : ${OUT_DIR}"
    printf "║  %-66s ║\n" "Metrics    : ${METRICS_CSV}"
    printf "║  %-66s ║\n" "Weights    : ${TOTALSEG_WEIGHTS_PATH}"
    printf "║  %-66s ║\n" "Checkpoint : ${CKPT_DIR}"
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
    echo "║  TotalSegmentator  —  ZERO-SHOT  —  COMPLETE                    ║"
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
# Phase 1 — Verify test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Verify test set symlink"
echo "================================================================"
if [[ ! -d "${IN_DIR}" ]]; then
    echo "ERROR: imagesTs not found at ${IN_DIR}"
    echo "  Run: bash data/setup_symlinks.sh <raw_dataset_dir>"
    exit 1
fi
N_CASES=$(ls "${IN_DIR}"/*.nii.gz 2>/dev/null | wc -l)
echo "  Found ${N_CASES} test cases in ${IN_DIR}"
echo "[OK] Phase 1"

# ─────────────────────────────────────────────
# Phase 2 — Zero-shot inference
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 2: TotalSegmentator zero-shot inference"
echo "  Task   : heartchambers_highres"
echo "  Input  : ${IN_DIR}"
echo "  Output : ${OUT_DIR}"
echo "================================================================"
if is_done "p2_inference"; then
    echo "[SKIP] Phase 2: inference already done"
else
    python models/totalsegmentator/run_zeroshot.py \
        --input-dir  "${IN_DIR}" \
        --output-dir "${OUT_DIR}"
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
