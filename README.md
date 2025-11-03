# RetBio-Kidney-DL: Retinal Biomarker Deep Learning for Kidney Function and Disease Classification

**Supplementary Repository — eGFR Regression & CKD Classification**

This repository is part of the *RetBio-Kidney-DL* research program — a suite of reproducible deep learning experiments exploring retinal biomarkers as non-invasive indicators of kidney health.

🧠 **Scope of this repository**
- **Task 1 — eGFR Regression:** Predict estimated glomerular filtration rate (eGFR) as a continuous outcome from retinal fundus images.  
- **Task 2 — CKD Binary Classification:** Classify presence or absence of chronic kidney disease (CKD / ESRD).  

> The survival (time-to-ESRD) component is maintained separately at  
> 🔗 [**retina-esrd-survival**](https://github.com/Effendy77/retina-esrd-survival)

---

## 🚀 Key Features
- End-to-end **PyTorch / timm** training pipeline  
- Compatible with **RETFound** and other pretrained vision transformers  
- Unified interface for regression & classification tasks  
- Optional fusion of retinal vascular metrics (CRAE, CRVE, AVR)  
- Lightweight **dummy dataset** for smoke testing and CI  

---

## 🧩 Installation

### Conda
```bash
conda env create -f environment.yml
conda activate retbio-kidney-dl

pip / venv
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

📂 Data Schema

Each CSV row corresponds to one retinal image.

Column	Description
image_path	relative or absolute path to the image
age, sex	optional covariates
egfr	continuous target for regression
ckd_label / esrd_label	binary label (0 = no CKD, 1 = CKD/ESRD)
crae, crve, avr	optional vascular features concatenated to embeddings
🧠 Quick Smoke Tests (dummy_data/)

Task 1 — eGFR Regression
python -m src.train \
  --task regression \
  --train_csv dummy_data/metadata.csv \
  --val_csv dummy_data/metadata.csv \
  --target egfr \
  --epochs 1 --batch_size 2
Task 2 — CKD Classification
python -m src.train \
  --task classification \
  --train_csv dummy_data/metadata.csv \
  --val_csv dummy_data/metadata.csv \
  --target ckd_label \
  --epochs 1 --batch_size 2

🧪 Example Training Commands
eGFR Regression
python -m src.train \
  --task regression \
  --train_csv data/ukb/train.csv \
  --val_csv data/ukb/val.csv \
  --target egfr \
  --img_size 448 --epochs 30 --batch_size 32 \
  --outdir outputs/egfr_run

CKD Classification
python -m src.train \
  --task classification \
  --train_csv data/ukb/train.csv \
  --val_csv data/ukb/val.csv \
  --target ckd_label \
  --img_size 448 --epochs 20 --batch_size 32 \
  --outdir outputs/ckd_run


🧠 Using RETFound or Other Pretrained Backbones
--backbone retfound_resnet50 \
--checkpoint /path/to/retfound.ckpt


📈 Outputs
After training, each run directory (e.g. outputs/egfr_run/) contains:
best.pt                # model weights
metrics.csv            # per-epoch metrics
config_snapshot.json   # reproducibility log


📊 Relationship to the Full RetBio-Kidney-DL Framework
RepositoryFocusDescriptionretbio-kidney-dl (this repo)eGFR regression & CKD classificationPredict kidney function and disease from retinal imagesretina-esrd-survivalTime-to-ESRD survivalCox-based survival prediction of kidney failure
Together, these repositories form the computational backbone for the RetBio-Kidney-DL publication.

🧾 Citation
If you use this codebase, please cite:
@software{retbio_kidney_dl_2025,
  author  = {Effendy Bin Hashim and collaborators},
  title   = {RetBio-Kidney-DL: Deep learning of retinal biomarkers for kidney function and disease},
  year    = {2025},
  url     = {https://github.com/Effendy77/retina-renal-ai}
}


👩‍🔬 Authors & Acknowledgements
Developed by Effendy Bin Hashim
PhD Researcher, Department of Eye and Vision Science, University of Liverpool
Supervisors: Prof Yalin Zheng · Prof Gregory Y.H. Lip
Affiliated with Liverpool Heart and Chest Hospital and Institute of Life Course & Medical Sciences

🧭 License
Open-source for academic and non-commercial use under the MIT License.




