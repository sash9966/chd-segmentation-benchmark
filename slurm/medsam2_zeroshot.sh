#!/usr/bin/env bash
#SBATCH --job-name=medsam2-zeroshot
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

module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/miniconda/envs/medsam2_env

export PYTHONUNBUFFERED=1

REPO=/scratch/users/sastocke/chd-segmentation-benchmark
cd ${REPO}

echo "=== MedSAM2 zero-shot ==="
echo "Input:  ${REPO}/data/imagesTs"
echo "Output: ${REPO}/results/medsam2/zeroshot"

python models/medsam/run_zeroshot.py \
    --input-dir   data/imagesTs \
    --output-dir  results/medsam2/zeroshot \
    --medsam2-dir models/medsam/MedSAM2 \
    --checkpoint  models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt \
    --device cuda

echo ""
echo "=== Computing metrics ==="
python evaluation/compute_metrics.py \
    --pred-dir    results/medsam2/zeroshot \
    --gt-dir      data/labelsTs \
    --output-csv  results/medsam2/zeroshot/metrics.csv \
    --splits-json chd_benchmark/splits/holdout.json

echo "Done. Metrics: ${REPO}/results/medsam2/zeroshot/metrics.csv"
