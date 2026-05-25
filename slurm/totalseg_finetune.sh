#!/usr/bin/env bash
#SBATCH --job-name=totalseg-finetune
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

module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/miniconda/envs/totalseg_env

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export TOTALSEG_WEIGHTS_PATH=/scratch/users/sastocke/chd-segmentation-benchmark/models/totalsegmentator/weights
export PYTHONUNBUFFERED=1

REPO=/scratch/users/sastocke/chd-segmentation-benchmark
FOLD=0
DATASET_ID=30

cd ${REPO}

echo "=== TotalSegmentator fine-tune (fold ${FOLD}) ==="

# Import TS heartchambers_highres weights into Dataset030 experiment
python models/totalsegmentator/fine_tune.py \
    --dataset-id ${DATASET_ID} \
    --fold       ${FOLD} \
    --task       heartchambers_highres \
    --input-dir  data/imagesTs \
    --output-dir results/totalsegmentator/finetuned

echo ""
echo "=== Computing metrics ==="
python evaluation/compute_metrics.py \
    --pred-dir    results/totalsegmentator/finetuned \
    --gt-dir      data/labelsTs \
    --output-csv  results/totalsegmentator/finetuned/metrics.csv \
    --splits-json chd_benchmark/splits/holdout.json

echo "Done. Metrics: ${REPO}/results/totalsegmentator/finetuned/metrics.csv"
