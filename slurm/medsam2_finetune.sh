#!/usr/bin/env bash
#SBATCH --job-name=medsam2-finetune
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

module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2

source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/miniconda/envs/medsam2_env

export PYTHONUNBUFFERED=1

REPO=/scratch/users/sastocke/chd-segmentation-benchmark
cd ${REPO}

echo "=== MedSAM2 fine-tune: all 7 CHD classes ==="
echo "Splits: chd_benchmark/splits/holdout.json (fold 0 train/val)"

for CLASS_ID in 1 2 3 4 5 6 7; do
    echo ""
    echo "  --- Class ${CLASS_ID} ---"
    python models/medsam/fine_tune.py \
        --splits-json   chd_benchmark/splits/holdout.json \
        --images-dir    data/imagesTr \
        --labels-dir    data/labelsTr \
        --medsam2-dir   models/medsam/MedSAM2 \
        --checkpoint    models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt \
        --output-dir    models/medsam/checkpoints/finetuned \
        --class-id      ${CLASS_ID} \
        --device cuda
done

echo ""
echo "=== MedSAM2 fine-tuned inference ==="
python models/medsam/run_finetuned.py \
    --input-dir       data/imagesTs \
    --output-dir      results/medsam2/finetuned \
    --medsam2-dir     models/medsam/MedSAM2 \
    --checkpoint-dir  models/medsam/checkpoints/finetuned \
    --base-checkpoint models/medsam/MedSAM2/checkpoints/MedSAM2_latest.pt \
    --device cuda

echo ""
echo "=== Computing metrics ==="
python evaluation/compute_metrics.py \
    --pred-dir    results/medsam2/finetuned \
    --gt-dir      data/labelsTs \
    --output-csv  results/medsam2/finetuned/metrics.csv \
    --splits-json chd_benchmark/splits/holdout.json

echo "Done. Metrics: ${REPO}/results/medsam2/finetuned/metrics.csv"
