# 📌 Artifact Severity Scoring

This repository contains the code and pretrained weights accompanying our [MIDL 2025 paper](https://2025.midl.io) titled:  
**_An Unsupervised Approach for Artifact Severity Scoring in Multi-Contrast MR Images_** [LINK](https://openreview.net/forum?id=73GUgAhllx#discussion)

We present an unsupervised deep learning method for quantifying artifact severity in clinical multi-contrast MR scans, particularly in the presence of noise, motion, and ghosting artifacts. Our model predicts a scalar severity score per image or slice, enabling automated quality control without manual annotations.

---

## 🚀 Getting Started

### 🧠 Prerequisites
- Python 3.10
- PyTorch ≥ 1.13
- Git LFS (for model weight downloads)
- See `requirements.txt` for full package versions

You can install the dependencies and clone the repo:

```bash
git clone https://github.com/shays15/artifact_scoring.git
cd artifact_scoring
pip install -r requirements.txt
```

### ⚙️ Setup

Download model weights via Git LFS (if not automatically downloaded):

```bash
git lfs install
git lfs pull
```

---

## 🏋️ Training

To train the model on your own dataset:

```bash
python train.py \
  --dataset /path/to/dataset \
  --gpu 0 \
  --exp_name my_experiment
```

- `--dataset`: Path to the directory of training/validation slices  
- `--gpu`: GPU device ID  
- `--exp_name`: Name of the experiment output directory (logs, checkpoints)

_Note: Training was performed on a private dataset that cannot be released._

---

## 🔍 Testing

To generate an artifact severity score for a single 3D NIfTI scan:

```bash
python test.py \
  --nifti-path /path/to/image.nii.gz \
  --gpu 0
```

The script will extract the central 60% of slices, apply necessary preprocessing, and output an average scalar score indicating artifact severity.

---

## 📁 Repository Structure

```
artifact_scoring/
├── train.py              # Training script
├── test.py               # Slice-based inference and scoring
├── utils.py              # Model + transform definitions
├── requirements.txt
└── README.md
```

---

## 🧠 Citation

If you use this code, please cite our MIDL 2025 paper:

```
Hays, S.*, Remedios, S.*, Dewey, B.E., Prince, J.L., Landman, B.A., Pham, D.L., Newsome, S.D., Mowry, E.M. 
"An Unsupervised Approach for Artifact Severity Scoring in Multi-Contrast MR Images." 
Medical Imaging with Deep Learning (MIDL), 2025.
```

---

## 🙏 Acknowledgments

This material is partially supported by the **Johns Hopkins University Percy Pierre Fellowship** (Hays) and the **National Science Foundation Graduate Research Fellowship** under Grant No. **DGE-2139757** (Hays) and **DGE-1746891** (Remedios).

Development is partially supported by:

- **FG-2008-36966** (Dewey)  
- **CDMRP W81XWH2010912** (Prince)  
- **NIH R01 CA253923** and **R01 CA275015** (Landman)  
- **National MS Society RG-1507-05243** (Pham)  
- **PCORI MS-1610-37115** (Newsome & Mowry)

_The statements in this publication are solely the responsibility of the authors and do not necessarily represent the views of PCORI, its Board of Governors, or Methodology Committee._
