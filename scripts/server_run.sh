#!/usr/bin/env bash
# server_run.sh — pull latest code, set up data, and submit zero-shot jobs
#
# Run this from the repo root on Sherlock:
#   bash scripts/server_run.sh
#
# First-time only: copy and fill in your paths:
#   cp configs/paths_server.sh.example configs/paths_server.sh
#   nano configs/paths_server.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load server paths ──────────────────────────────────────────────────────────
PATHS_FILE="${REPO}/configs/paths_server.sh"
if [[ ! -f "$PATHS_FILE" ]]; then
    echo "ERROR: configs/paths_server.sh not found."
    echo "  cp configs/paths_server.sh.example configs/paths_server.sh"
    echo "  then fill in your Oak paths."
    exit 1
fi
source "$PATHS_FILE"

# ── Pull latest ────────────────────────────────────────────────────────────────
echo "=== git pull ==="
git -C "$REPO" pull

# ── First-time data setup ──────────────────────────────────────────────────────
SPLITS_JSON="${REPO}/chd_benchmark/splits/holdout.json"

if [[ ! -L "${REPO}/data/imagesTs" ]]; then
    echo ""
    echo "=== Setting up data symlinks ==="
    bash "${REPO}/data/setup_symlinks.sh" \
        "$NNUNET_RAW_DATASET" \
        "$NNUNET_PREPROCESSED"
else
    echo "=== Data symlinks already in place ==="
fi

if [[ ! -f "$SPLITS_JSON" ]]; then
    echo ""
    echo "=== Building benchmark splits ==="
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate base
    python "${REPO}/chd_benchmark/splits/build_splits.py" \
        --splits-final "${NNUNET_PREPROCESSED}/splits_final.json" \
        --images-ts-dir "${REPO}/data/imagesTs" \
        --output-dir "${REPO}/chd_benchmark/splits"
else
    echo "=== Splits already built ==="
fi

# ── Create results dirs ────────────────────────────────────────────────────────
mkdir -p "${REPO}/results/totalsegmentator/zeroshot"
mkdir -p "${REPO}/results/medsam2/zeroshot"
mkdir -p "${REPO}/slurm/logs"

# ── Submit zero-shot jobs ──────────────────────────────────────────────────────
echo ""
echo "=== Submitting zero-shot jobs ==="

TOTALSEG_JOB=$(sbatch --parsable "${REPO}/slurm/totalseg_zeroshot.sh")
echo "  TotalSegmentator  job ID: ${TOTALSEG_JOB}"

MEDSAM_JOB=$(sbatch --parsable "${REPO}/slurm/medsam2_zeroshot.sh")
echo "  MedSAM2           job ID: ${MEDSAM_JOB}"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Zero-shot jobs submitted                                        ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  TotalSegmentator  job ${TOTALSEG_JOB}"
echo "║  MedSAM2           job ${MEDSAM_JOB}"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Predictions →  results/totalsegmentator/zeroshot/              ║"
echo "║                 results/medsam2/zeroshot/                        ║"
echo "║  Metrics CSV →  results/totalsegmentator/zeroshot/metrics.csv   ║"
echo "║                 results/medsam2/zeroshot/metrics.csv             ║"
echo "║  SLURM logs  →  slurm/logs/totalseg_zeroshot_${TOTALSEG_JOB}.out"
echo "║                 slurm/logs/medsam2_zeroshot_${MEDSAM_JOB}.out"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Monitor:  squeue -u sastocke                                    ║"
echo "║  Compare:  python evaluation/summarize.py                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
