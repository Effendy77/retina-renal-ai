# retina-renal-ai (Monorepo)

One shared PyTorch pipeline for three tasks:
1) Task 1 — eGFR regression
2) Task 2 — CKD/ESRD binary classification
3) Task 3 — Time-to-event CKD/ESRD (Cox)

Backbone is a `timm` model (default `resnet50`, `num_classes=0`, global pooling). You can pass a local RetFound checkpoint via `--checkpoint`.

## Install
Conda:
    conda env create -f environment.yml
    conda activate retina-renal-ai

pip:
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Data schema
CSV fields:
- image_path (path relative to CSV)
- age (float), sex (M/F or 0/1)
Task columns:
- regression: egfr
- classification: ckd_label or esrd_label (0/1)
- survival: time (years), event (0/1)
Optional vascular features: crae, crve, avr (auto-concat to embeddings).

## Quick smoke tests (dummy data included)
Task 1:
    python -m src.train --task regression --train_csv dummy_data/metadata.csv --val_csv dummy_data/metadata.csv --target egfr --epochs 1 --batch_size 2
Task 2:
    python -m src.train --task classification --train_csv dummy_data/metadata.csv --val_csv dummy_data/metadata.csv --target ckd_label --epochs 1 --batch_size 2
Task 3:
    python -m src.train --task survival --train_csv dummy_data/metadata_surv.csv --val_csv dummy_data/metadata_surv.csv --target time --event_col event --epochs 1 --batch_size 2

## Train on your data
Regression example:
    python -m src.train --task regression --train_csv data/ukb/train.csv --val_csv data/ukb/val.csv --target egfr --img_size 448 --epochs 30 --batch_size 32 --outdir outputs/egfr_run
Classification example:
    python -m src.train --task classification --train_csv data/ukb/train.csv --val_csv data/ukb/val.csv --target ckd_label --img_size 448 --epochs 20 --batch_size 32 --outdir outputs/ckd_run
Survival example:
    python -m src.train --task survival --train_csv data/ukb/train_surv.csv --val_csv data/ukb/val_surv.csv --target time --event_col event --img_size 448 --epochs 20 --batch_size 32 --outdir outputs/ckd_surv

## Optional RetFound checkpoint
Use:
    --backbone retfound_resnet50 --checkpoint /path/to/retfound.ckpt

## Outputs
- outputs/.../best.pt, outputs/.../config_snapshot.json
