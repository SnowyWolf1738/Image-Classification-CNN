# Image Classification CNN

> A PyTorch-based computer vision system for **robust image classification under distribution shift**, featuring explicit handling of out-of-domain (OOD) inputs via a multi-head architecture.

---

## Overview
This project implements a **three-headed convolutional neural network** designed to:

- Accurately classify **in-domain (ID)** images across 10 categories  
- Detect whether an input is **out-of-domain (OOD)**  
- Produce **controlled uncertainty (near-random predictions)** for OOD samples  

The system was built to explore **robust image classification under distribution shift**, combining architectural design, data-centric strategies, and performance optimization. OOD images are purposely misclasiffied so that the neural network cannot be misused for images outside of it's specific use case.

---

## Features
- **Three-head architecture** for domain detection, classification, and uncertainty modeling  
- **Explicit OOD handling** via KL divergence to a uniform distribution  
- **Data-centric pipeline** for handling class imbalance and image variability  
- **GPU-accelerated augmentations** using Kornia  
- **Optimized training pipeline** with smart caching and CUDA acceleration  

---

## Tech Stack

**Core:**  
- Python, PyTorch (Experimented with Tensorflow in early design, but PyTorch was selected for its flexibility in implementing custom architectures.)

**Libraries:**  
- torchvision  
- Kornia (GPU augmentations)  

**Hardware:**  
- CUDA (GPU acceleration, tested on Google Colab T4)

---

## Key Technical Highlights

### 1. Data-Centric Design
- Identified **class imbalance** (e.g., 51 vs 99 samples per class)
- Handled **extreme variation in image size and aspect ratio**
- Designed preprocessing pipeline with:
  - Resampling strategies  
  - Standardized resizing (224×224)  
  - Targeted augmentation pipelines  

---

## 2. Data Pipeline

- Standardized resizing: **224×224**
- Resampling to mitigate class imbalance  
- Two augmentation regimes:
  - **Heavy (training):** crop, flip, rotation, color jitter, random erasing  
  - **Light (OOD/domain tasks):** rotation + flip  

---

### 3. Performance Optimization

#### CUDA Acceleration
- Leveraged GPU for model training and tensor operations  

#### Smart Image Caching
- Cached **resized images only** (not augmented)
- Preserved stochastic augmentation → improved accuracy from **<60% → ~70%**
- Cached images greatly increased training speed when retraining multiple times

#### GPU Augmentations (Kornia)
- Moved transformations from CPU → GPU  
- Eliminated preprocessing bottleneck  
- Increased training throughput significantly  

---

## Model Architecture

### Final Model: **Three-Head CNN**

#### 1. Domain Classifier Head
- Binary classifier (ID vs OOD)  
- Custom CNN  
- ~80% training accuracy  

#### 2. In-Domain Classifier Head
- **ResNet-18 (trained from scratch)**  
- 10-class classification (can be edited to include more or less classes)
- Best performer across all experiments  

#### 3. OOD Classifier Head
- Small CNN  
- Trained using **KL divergence to uniform distribution**  
- Produces near-random predictions for OOD inputs  

---

## Inference Pipeline

1. Predict **ID vs OOD** using domain head  
2. If **ID →** use ResNet-18 classifier  
3. If **OOD →** use OOD head (uniform predictions)  

This ensures:
- High-confidence predictions for known data  
- Calibrated uncertainty for unknown data  

---

## 🧪 Training Strategy

Each head is trained independently:

### Domain Head
- Data: ID + OOD  
- Augmentation: Light  
- Optimizer: Adam  
- Epochs: 50  

---

### In-Domain Head (ResNet-18)
- Data: ID only  
- Augmentation: Heavy  
- Optimizer: **SGD with momentum** (best performance)  
- Epochs: 110  
- Result:
  - ~95% training accuracy  
  - **70%+ evaluation accuracy**

---

### OOD Head
- Data: OOD only  
- Loss: KL divergence → uniform distribution  
- Goal: enforce uncertainty  

---

## Results

| Metric | Range |
|------|------|
| In-Domain Accuracy | **61% – 68%+** |
| OOD Accuracy | 25% – 45% |
| Training Time | ~20 minutes (Colab T4 GPU) |

---

## Project Structure
```bash
Image-Classifcation-CNN/
│── cnn.py
│── main.py
│── requirements.txt
│── README.md
```

---

## Installation & Setup

```bash
# Clone the repo
git clone https://github.com/SnowyWolf1738/Image-Classification-CNN.git

# Navigate into the folder
cd Image-Classification-CNN

# Install dependencies
pip install -r requirements.txt

# Move in-domain/out-domain training/classification images into the directory, then run the program
python main.py
```

If running on Google Colab, run the following command in a code block:
!pip install kornia
All other libraries should already be installed by default

---

## 📈 Key Learnings

- Larger models (**ResNet-18**) outperformed lightweight architectures when trained from scratch  
- Data augmentation must remain **stochastic** to be effective  
- GPU preprocessing can significantly reduce training bottlenecks  
- Handling OOD explicitly improves model reliability in real-world scenarios  

---

## 🔮 Future Improvements

- Add **pretrained backbone** (transfer learning)  
- Improve **OOD detection accuracy** with advanced methods (e.g., energy-based models)  
- Implement **calibration techniques** (temperature scaling)  
- Expand dataset for better generalization  

---

## 📬 Contact

**Your Name**  
- LinkedIn: https://linkedin.com/in/yourprofile  
- Email: your@email.com  

---

## ⭐ Why This Project Matters

This project demonstrates my ability to:

- Design **non-trivial deep learning architectures**
- Debug and optimize **training pipelines at scale**
- Apply **data-centric AI principles**
- Balance **accuracy, efficiency, and robustness**

It reflects a practical approach to **real-world ML problems**, especially where data distribution is unpredictable.