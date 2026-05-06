# 🛰️ Satellite Image Classification for Strategic Area Detection

> Binary classification of EuroSAT satellite imagery using 10 CNN architectures — comparing pretrained transfer learning models against custom CNNs.

---

## 👥 Authors

| Name | Role |
|------|------|
| **Sujal Krishna Das** | Project Lead, Pipeline Design, Dataset Preprocessing |
| **Pratham Shetty** | Pretrained Model Integration, Training Loop |
| **Rahul D Seervi** | Custom CNN Design, Lightweight Model Integration |
| **Shreyas Sajeev** | Visualizations, Result Analysis |

**Department of Artificial Intelligence and Robotics**

---

## 📋 Overview

This project implements a deep learning pipeline for **binary satellite image classification** using the [EuroSAT](https://github.com/phelber/EuroSAT) dataset. The goal is to detect **strategically significant land-use areas** — specifically Industrial zones and Highway corridors — from Sentinel-2 satellite imagery.

We benchmark **10 CNN architectures** under identical training conditions and compare their performance through multiple visualizations.

---

## 🗂️ Dataset

- **Source:** [EuroSAT (DFKI)](https://madm.dfki.de/files/sentinel/EuroSAT.zip)
- **Size:** 27,000 labeled Sentinel-2 satellite images across 10 classes
- **Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Input Size:** Resized to 128×128 RGB

### Binary Reclassification

| Label | Classes |
|-------|---------|
| `1` — Strategic | `Industrial`, `Highway` |
| `0` — Non-Strategic | All other 8 classes |

---

## 🏗️ Architecture

### Pretrained Models (Transfer Learning)
All pretrained models use **ImageNet weights** from `torchvision`, with the final classification layer replaced by `nn.Linear(in_features, 2)`.

| Model | Parameters | Modification |
|-------|-----------|--------------|
| ResNet18 | ~11M | `model.fc` |
| ResNet34 | ~21M | `model.fc` |
| VGG11 | ~133M | `model.classifier[6]` |
| AlexNet | ~61M | `model.classifier[6]` |
| DenseNet121 | ~8M | `model.classifier` |
| MobileNetV2 | ~3.4M | `model.classifier[1]` |
| EfficientNetB0 | ~5.3M | `model.classifier[1]` |
| ShuffleNet | ~2.3M | `model.fc` |

### Custom CNN Architecture
```
Conv2d(3→32, 3×3) → ReLU → MaxPool2d(2)
Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2)
Flatten → Linear(65536→128) → ReLU → Linear(128→2)
~0.5M parameters | No pretraining
```

---

## ⚙️ Training Setup

```python
Optimizer   : Adam (lr=1e-3)
Loss        : CrossEntropyLoss
Batch Size  : 32
Epochs      : 2
Device      : CUDA (GPU)
Train Split : 80% (21,600 images)
Test Split  : 20% (5,400 images)
```

---

## 📊 Results

| Model | Final Loss | Test Accuracy |
|-------|-----------|---------------|
| 🥇 **ShuffleNet** | **0.0398** | **99.35%** |
| 🥈 MobileNetV2 | 0.0476 | 99.00% |
| 🥉 EfficientNetB0 | 0.0386 | 98.65% |
| ResNet18 | 0.0820 | 96.17% |
| DenseNet121 | 0.0915 | 92.06% |
| Custom2 | ~0.25 | ~90.9% |
| Custom1 | ~0.49 | ~90.6% |
| ResNet34 | 0.1280 | 90.44% |
| AlexNet | 0.3201 | 88.69% |
| VGG11 | 0.4823 | 81.31% |

> **Key Finding:** ShuffleNet (2.3M params) outperformed VGG11 (133M params) by **18 percentage points**, demonstrating that lightweight efficient architectures are superior for rapid satellite image fine-tuning.

---

## 📈 Visualizations

The notebook generates 6 plots:

1. **Loss Comparison** — Overlaid loss curves for all 10 models across epochs
2. **Accuracy Bar Chart** — Test accuracy comparison across all models
3. **Best Model Highlight** — Isolated bar for the top-performing model (ShuffleNet)
4. **Loss Curves (All Models)** — Full subplot of all convergence trajectories
5. **Model Ranking** — Horizontal bar chart sorted by accuracy (descending)
6. **Accuracy Pie Chart** — Proportional accuracy distribution across all models

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Install Dependencies
```bash
pip install torch torchvision pillow matplotlib
```

### 3. Run on Google Colab (Recommended)
Open the notebook in Colab for free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

The notebook will automatically:
- Download and extract the EuroSAT dataset
- Detect the correct dataset path
- Train all 10 models on GPU
- Generate all visualizations

### 4. Run Locally
```bash
jupyter notebook kdrproject.ipynb
```

---

## 📁 Project Structure

```
📦 repo-root
 ┣ 📓 kdrproject.ipynb       # Main Colab notebook
 ┣ 📄 README.md              # This file
 ┣ 📂 2750/                  # EuroSAT dataset (auto-downloaded)
 ┃ ┣ 📂 AnnualCrop/
 ┃ ┣ 📂 Forest/
 ┃ ┣ 📂 Highway/
 ┃ ┣ 📂 Industrial/
 ┃ ┗ 📂 ...
```

---

## 🔑 Key Classes

### `SafeImageFolder`
Subclass of `torchvision.datasets.ImageFolder` that forces `.convert("RGB")` on all images, preventing channel mismatch errors from multi-band satellite patches.

### `DefenseDataset`
Binary reclassification wrapper — dynamically maps `Industrial` and `Highway` → label `1` (strategic), all other classes → label `0` (non-strategic).

### `modify_model(model, name)`
Generic function that replaces each model's output head with `nn.Linear(in_features, 2)` for binary classification, using architecture-specific targeting.

---

## 💡 Insights

- **Lightweight wins:** ShuffleNet, MobileNetV2, and EfficientNetB0 — all under 6M parameters — dominated the leaderboard
- **Size ≠ Accuracy:** VGG11 (133M params) achieved the worst pretrained accuracy (81.31%) due to convergence failure at 2 epochs
- **Transfer learning is powerful:** All pretrained models except VGG11/AlexNet converged in epoch 1; custom CNNs required more epochs
- **Custom CNNs are viable:** ~0.5M params, no pretraining → ~91% accuracy on 21,600 training samples

---

## 🔭 Future Work

- [ ] Extended training (5–10 epochs) with learning rate scheduling
- [ ] Data augmentation (random flips, rotations, colour jitter)
- [ ] Focal loss to handle class imbalance (~18% strategic samples)
- [ ] Multi-class extension beyond binary classification
- [ ] Multi-temporal satellite sequences for change detection
- [ ] Model ensembling for production deployment

---

## 📜 License

This project is for academic purposes. The EuroSAT dataset is published under the [MIT License](https://github.com/phelber/EuroSAT/blob/master/LICENSE).

---

## 📚 References

1. Helber et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark.* IEEE JSTARS.
2. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
3. Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
4. Ma et al. (2018). *ShuffleNet V2: Practical Guidelines for Efficient CNN Design.* ECCV.
5. Howard et al. (2017). *MobileNets: Efficient CNNs for Mobile Vision.* arXiv.
