#!/usr/bin/env bash
#SBATCH --job-name=medsam2_zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/medsam2_zeroshot_%j.out
#SBATCH --error=slurm/logs/medsam2_zeroshot_%j.err

set -euo pipefail

SCRATCH=/scratch/users/sastocke
REPO=${SCRATCH}/chd-segmentation-benchmark

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/miniconda/envs/medsam2_env

cd ${REPO}

python models/medsam/run_zeroshot.py \
    --input-dir   data/imagesTs \
    --output-dir  results/medsam2/zeroshot \
    --medsam2-dir models/medsam/MedSAM2 \
    --checkpoint  models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt \
    --device cuda

echo "Done — evaluating..."

python evaluation/compute_metrics.py \
    --pred-dir   results/medsam2/zeroshot \
    --gt-dir     data/labelsTs \
    --output-csv results/medsam2/zeroshot/metrics.csv \
    --splits-json chd_benchmark/splits/holdout.json
