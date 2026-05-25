#!/bin/bash
# =============================================================================
#  totalseg_finetune.sh
#  TotalSegmentator v2 — fine-tune on Dataset030_imageCHD_HU (fold 0)
#
#  Steps
#    Phase 1: Verify data + splits
#    Phase 2: Import TotalSegmentator pretrained weights into nnUNet experiment
#    Phase 3: nnUNetv2_train (resumes from checkpoint if interrupted)
#    Phase 4: Inference on imagesTs
#    Phase 5: Compute Dice + HD95 metrics
#
#  RESUME SUPPORT
#    Each phase creates a .done marker in CKPT_DIR.
#    nnUNet itself resumes from its own latest checkpoint if training
#    was interrupted mid-epoch — no extra handling needed.
# =============================================================================
#SBATCH --job-name=totalseg-ft
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/totalseg_finetune_%j.out
#SBATCH --error=/scratch/users/sastocke/chd-segmentation-benchmark/slurm/logs/totalseg_finetune_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/miniconda/envs/totalseg_env
hash -r

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export TOTALSEG_WEIGHTS_PATH="/scratch/users/sastocke/chd-segmentation-benchmark/models/totalsegmentator/weights"
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────
# 2.  Configuration
# ─────────────────────────────────────────────
DATASET_ID=30
DATASET_NAME="Dataset030_imageCHD_HU"
FOLD=0
TS_TASK="heartchambers_highres"
REPO="/scratch/users/sastocke/chd-segmentation-benchmark"
IN_DIR="${REPO}/data/imagesTs"
OUT_DIR="${REPO}/results/totalsegmentator/finetuned"
METRICS_CSV="${OUT_DIR}/metrics.csv"
SPLITS_JSON="${REPO}/chd_benchmark/splits/holdout.json"
CKPT_DIR="${REPO}/results/totalsegmentator/.checkpoints/finetune"
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
    echo "║  TotalSegmentator  —  FINE-TUNE  —  START                       ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "TS Task    : ${TS_TASK}  (pretrained weights)"
    printf "║  %-66s ║\n" "Fold       : ${FOLD}  (from holdout.json)"
    printf "║  %-66s ║\n" "Input      : ${IN_DIR}"
    printf "║  %-66s ║\n" "Output     : ${OUT_DIR}"
    printf "║  %-66s ║\n" "Metrics    : ${METRICS_CSV}"
    printf "║  %-66s ║\n" "nnUNet raw : ${nnUNet_raw}"
    printf "║  %-66s ║\n" "nnUNet res : ${nnUNet_results}"
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
    echo "║  TotalSegmentator  —  FINE-TUNE  —  COMPLETE                    ║"
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
# Phase 1 — Verify data + splits
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Verify data symlinks and splits"
echo "================================================================"
if [[ ! -d "${REPO}/data/imagesTr" ]]; then
    echo "ERROR: imagesTr not found. Run: bash data/setup_symlinks.sh"
    exit 1
fi
if [[ ! -f "${SPLITS_JSON}" ]]; then
    echo "ERROR: splits not found at ${SPLITS_JSON}"
    echo "  Run: python chd_benchmark/splits/build_splits.py"
    exit 1
fi
N_TR=$(ls "${REPO}/data/imagesTr"/*.nii.gz 2>/dev/null | wc -l)
N_TS=$(ls "${REPO}/data/imagesTs"/*.nii.gz 2>/dev/null | wc -l)
echo "  Train: ${N_TR} cases | Test: ${N_TS} cases"
echo "[OK] Phase 1"

# ─────────────────────────────────────────────
# Phase 2 — Import TotalSegmentator weights
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 2: Import TotalSegmentator pretrained weights"
echo "  Task      : ${TS_TASK}"
echo "  Dataset ID: ${DATASET_ID}"
echo "================================================================"
if is_done "p2_import_weights"; then
    echo "[SKIP] Phase 2: weights already imported"
else
    totalseg_import_weights -d ${DATASET_ID} -t ${TS_TASK}
    mark_done "p2_import_weights"
    echo "[OK] Phase 2"
fi

# ─────────────────────────────────────────────
# Phase 3 — Fine-tune training
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 3: nnUNetv2_train fine-tune  (fold ${FOLD})"
echo "  Dataset    : ${DATASET_NAME}"
echo "  Config     : 3d_fullres"
echo "  nnUNet raw : ${nnUNet_raw}"
echo "  nnUNet res : ${nnUNet_results}"
echo "================================================================"
if is_done "p3_train_fold${FOLD}"; then
    echo "[SKIP] Phase 3: training fold ${FOLD} already done"
else
    # --c flag lets nnUNet resume from its own checkpoint if interrupted
    nnUNetv2_train ${DATASET_ID} 3d_fullres ${FOLD} \
        -tr nnUNetTrainer --c
    mark_done "p3_train_fold${FOLD}"
    echo "[OK] Phase 3"
fi

# ─────────────────────────────────────────────
# Phase 4 — Inference on test set
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 4: Inference on test set"
echo "  Input  : ${IN_DIR}"
echo "  Output : ${OUT_DIR}"
echo "================================================================"
if is_done "p4_inference"; then
    echo "[SKIP] Phase 4: inference already done"
else
    nnUNetv2_predict \
        -i "${IN_DIR}" \
        -o "${OUT_DIR}" \
        -d ${DATASET_ID} \
        -c 3d_fullres \
        -f ${FOLD} \
        -tr nnUNetTrainer
    mark_done "p4_inference"
    echo "[OK] Phase 4"
fi

# ─────────────────────────────────────────────
# Phase 5 — Metrics
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Phase 5: Compute Dice + HD95 metrics"
echo "  Predictions : ${OUT_DIR}"
echo "  Ground truth: ${REPO}/data/labelsTs"
echo "  Output CSV  : ${METRICS_CSV}"
echo "================================================================"
if is_done "p5_metrics"; then
    echo "[SKIP] Phase 5: metrics already computed"
else
    python evaluation/compute_metrics.py \
        --pred-dir    "${OUT_DIR}" \
        --gt-dir      data/labelsTs \
        --output-csv  "${METRICS_CSV}" \
        --splits-json "${SPLITS_JSON}"
    mark_done "p5_metrics"
    echo "[OK] Phase 5"
fi

print_footer
