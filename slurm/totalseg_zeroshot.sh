#!/usr/bin/env bash
#SBATCH --job-name=totalseg_zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/totalseg_zeroshot_%j.out
#SBATCH --error=slurm/logs/totalseg_zeroshot_%j.err

set -euo pipefail

SCRATCH=/scratch/users/sastocke
REPO=${SCRATCH}/chd-segmentation-benchmark

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate ${SCRATCH}/miniconda/envs/totalseg_env

# Weights stored in repo on scratch (auto-downloaded on first run)
export TOTALSEG_WEIGHTS_PATH=${REPO}/models/totalsegmentator/weights

cd ${REPO}

python models/totalsegmentator/run_zeroshot.py \
    --input-dir  data/imagesTs \
    --output-dir results/totalsegmentator/zeroshot

echo "Done — evaluating..."

python evaluation/compute_metrics.py \
    --pred-dir   results/totalsegmentator/zeroshot \
    --gt-dir     data/labelsTs \
    --output-csv results/totalsegmentator/zeroshot/metrics.csv \
    --splits-json chd_benchmark/splits/holdout.json
