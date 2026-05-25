#!/usr/bin/env bash
# server_run.sh — git pull, set up data, and submit all 4 comparison jobs:
#   totalseg  zero-shot  │  medsam2  zero-shot
#   totalseg  fine-tune  │  medsam2  fine-tune
#
# Usage (from repo root on Sherlock):
#   bash scripts/server_run.sh
#
# First time only — copy and fill in your paths:
#   cp configs/paths_server.sh.example configs/paths_server.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load server paths ──────────────────────────────────────────────────────────
PATHS_FILE="${REPO}/configs/paths_server.sh"
if [[ ! -f "$PATHS_FILE" ]]; then
    echo "ERROR: configs/paths_server.sh not found."
    echo "  cp configs/paths_server.sh.example configs/paths_server.sh"
    exit 1
fi
source "$PATHS_FILE"

# ── Pull latest ────────────────────────────────────────────────────────────────
echo "=== git pull ==="
cd "$REPO" && git pull

# ── First-time data setup ──────────────────────────────────────────────────────
if [[ ! -L "${REPO}/data/imagesTs" ]]; then
    echo ""
    echo "=== Setting up data symlinks ==="
    bash "${REPO}/data/setup_symlinks.sh" \
        "${NNUNET_RAW_DATASET}" \
        "${NNUNET_PREPROCESSED}"
else
    echo "=== Data symlinks OK ==="
fi

if [[ ! -f "${REPO}/chd_benchmark/splits/holdout.json" ]]; then
    echo ""
    echo "=== Building benchmark splits from ${DATASET_NAME} ==="
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate base
    python "${REPO}/chd_benchmark/splits/build_splits.py" \
        --splits-final   "${NNUNET_PREPROCESSED}/splits_final.json" \
        --images-ts-dir  "${REPO}/data/imagesTs" \
        --output-dir     "${REPO}/chd_benchmark/splits"
else
    echo "=== Splits OK (${REPO}/chd_benchmark/splits/holdout.json) ==="
fi

# ── Ensure output dirs exist ───────────────────────────────────────────────────
mkdir -p \
    "${REPO}/results/totalsegmentator/zeroshot" \
    "${REPO}/results/totalsegmentator/finetuned" \
    "${REPO}/results/medsam2/zeroshot" \
    "${REPO}/results/medsam2/finetuned" \
    "${REPO}/slurm/logs"

# ── Submit jobs ────────────────────────────────────────────────────────────────
echo ""
echo "=== Submitting 4 jobs ==="

TSEG_ZS=$(sbatch --parsable "${REPO}/slurm/totalseg_zeroshot.sh")
echo "  [${TSEG_ZS}]  TotalSegmentator  zero-shot"

MSAM_ZS=$(sbatch --parsable "${REPO}/slurm/medsam2_zeroshot.sh")
echo "  [${MSAM_ZS}]  MedSAM2           zero-shot"

TSEG_FT=$(sbatch --parsable "${REPO}/slurm/totalseg_finetune.sh")
echo "  [${TSEG_FT}]  TotalSegmentator  fine-tune"

MSAM_FT=$(sbatch --parsable "${REPO}/slurm/medsam2_finetune.sh")
echo "  [${MSAM_FT}]  MedSAM2           fine-tune"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Jobs submitted for Dataset030_imageCHD_HU                          ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
printf "║  %-6s  TotalSegmentator  zero-shot\n" "[${TSEG_ZS}]"
printf "║  %-6s  MedSAM2           zero-shot\n" "[${MSAM_ZS}]"
printf "║  %-6s  TotalSegmentator  fine-tune  (48h)\n" "[${TSEG_FT}]"
printf "║  %-6s  MedSAM2           fine-tune  (24h)\n" "[${MSAM_FT}]"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Predictions                                                         ║"
echo "║    results/totalsegmentator/{zeroshot,finetuned}/*.nii.gz            ║"
echo "║    results/medsam2/{zeroshot,finetuned}/*.nii.gz                     ║"
echo "║  Metrics CSV                                                          ║"
echo "║    results/totalsegmentator/{zeroshot,finetuned}/metrics.csv         ║"
echo "║    results/medsam2/{zeroshot,finetuned}/metrics.csv                  ║"
echo "║  SLURM logs                                                           ║"
echo "║    slurm/logs/totalseg_zeroshot_${TSEG_ZS}.out                       ║"
echo "║    slurm/logs/medsam2_zeroshot_${MSAM_ZS}.out                        ║"
echo "║    slurm/logs/totalseg_finetune_${TSEG_FT}.out                       ║"
echo "║    slurm/logs/medsam2_finetune_${MSAM_FT}.out                        ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Monitor:  squeue -u sastocke                                         ║"
echo "║  Compare:  python evaluation/summarize.py                             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
