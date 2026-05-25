# CHD Segmentation Model Comparison

Stanford — Congenital Heart Disease (CHD) benchmark comparing pretrained and fine-tuned segmentation models.

**Models compared:** TotalSegmentator v2 · MedSAM · MONAI Auto3DSeg  
**Protocol:** zero-shot → fine-tune on CHD train split → evaluate on held-out test set  
**Dataset:** ImageCHD (8 classes: background + LV, RV, LA, RA, myocardium, aorta, PA)  
**Data source:** nnUNet v2 layout — symlinked, never duplicated.

---

## Repository layout

```
configs/                  dataset paths (fill in before running)
data/                     symlinks to nnUNet raw + preprocessed data
  setup_symlinks.sh       creates symlinks; run once per machine
chd_benchmark/
  data_qa.py              NIfTI header + label QA
  splits/
    build_splits.py       convert splits_final.json → benchmark JSONs
    holdout.json          train/val/test split (generated)
    folds.json            all CV folds (generated)
models/
  totalsegmentator/       zero-shot + nnUNet fine-tune wrappers
  medsam/                 zero-shot + per-class fine-tune + finetuned inference
  autoseg/                Auto3DSeg train-from-scratch + zero-shot
evaluation/
  compute_metrics.py      Dice + HD95 per case per class → CSV
  summarize.py            aggregate all CSVs → comparison table
environments/             conda env YAML files (one per model)
scripts/
  01_setup_data.sh        symlink + splits + QA
  02_run_zeroshot.sh      zero-shot inference all models
  03_run_finetune.sh      fine-tune + inference one model at a time
  04_evaluate.sh          metrics + comparison table
results/                  gitignored — written at runtime
```

---

## Quickstart

### 1. Clone and configure

```bash
git clone <your-remote> && cd ComparisonModels
# Edit configs/dataset.yaml with your nnUNet paths
```

### 2. Set up data (once per machine)

```bash
bash scripts/01_setup_data.sh \
    /path/to/nnUNet_raw/Dataset001_CHD \
    /path/to/nnUNet_preprocessed/Dataset001_CHD \
    /path/to/nnUNet_preprocessed/Dataset001_CHD/splits_final.json
```

This creates symlinks in `data/`, builds `chd_benchmark/splits/holdout.json`, and runs QA.

### 3. Create environments

```bash
conda env create -f environments/totalsegmentator.yml
conda env create -f environments/medsam.yml
conda env create -f environments/autoseg.yml
```

See each YAML for additional manual steps (checkpoint downloads, source installs).

### 4. Zero-shot runs

```bash
conda activate totalseg
bash scripts/02_run_zeroshot.sh totalsegmentator

conda activate medsam
bash scripts/02_run_zeroshot.sh medsam

conda activate autoseg
bash scripts/02_run_zeroshot.sh autoseg
```

### 5. Fine-tune runs

```bash
conda activate totalseg && bash scripts/03_run_finetune.sh totalsegmentator 1 0
conda activate medsam   && bash scripts/03_run_finetune.sh medsam
conda activate autoseg  && bash scripts/03_run_finetune.sh autoseg
```

### 6. Evaluate all

```bash
# Any env with nibabel/numpy/pandas/scipy works
bash scripts/04_evaluate.sh all
# Results written to results/comparison_table.csv
```

---

## Data layout expected (nnUNet v2)

```
nnUNet_raw/Dataset001_CHD/
    imagesTr/   case_0000.nii.gz, ...   (training images)
    labelsTr/   case.nii.gz, ...        (training labels)
    imagesTs/   case_0000.nii.gz, ...   (test images)
    labelsTs/   case.nii.gz, ...        (test labels — for eval only)
    dataset.json
nnUNet_preprocessed/Dataset001_CHD/
    splits_final.json
    nnUNetPlans.json
    ...
```

Symlinks in `data/` point here. No data is duplicated.

---

## Label map

| ID | Structure |
|----|-----------|
| 0  | background |
| 1  | left ventricle |
| 2  | right ventricle |
| 3  | left atrium |
| 4  | right atrium |
| 5  | myocardium |
| 6  | aorta |
| 7  | pulmonary artery |

---

## Adding a new model

1. Create `models/<name>/run_zeroshot.py` and `models/<name>/fine_tune.py`
2. Add an `environments/<name>.yml`
3. Add a case in `scripts/02_run_zeroshot.sh` and `scripts/03_run_finetune.sh`
4. The evaluation scripts auto-discover `results/<name>/<variant>/metrics.csv`
