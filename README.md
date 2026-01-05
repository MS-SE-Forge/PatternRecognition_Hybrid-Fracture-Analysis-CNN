# Hybrid Fracture Analysis System 🦴

> **Part 2 Submission**
> **Group Number**: [Insert Group Number]
> **Course**: Pattern Recognition (M.Sc.)

## 👥 Group Members & Roles
| Student Name | Role | Responsibilities |
| :--- | :--- | :--- |
| **Kazeem Asiwaju-Bello** | **Technical Lead** | Model implementation, Experiments, RQ definition, Code submission. |
| **OluwaTosin Ojo** | **Figures & Pres.** | Figure design, Visualization, Presentation slides. |
| **Priyanka Mohan** | **Report Lead** | Report writing, Storytelling, Narrative coherence. |

---

## 📚 Project Overview
This project implements a **Hybrid Fracture Analysis System** that combines Deep Learning (CNNs) with Rule-Based Morphological Analysis to detect bone fractures and estimate their severity (Hairline vs. Severe).

### Key Features

1.  **Deep Learning Classifier**: ResNet50 (and ResNet18 comparison) to detect fractures.
2.  **Rule-Based Logic**: Uses Edge Detection and Hough Transform to measure displacement.
3.  **Hybrid Reasoning**: Combines AI confidence with measured displacement for final diagnosis.


---

## 🧠 Model Architecture
We investigate multiple architectures to find the best balance of speed and accuracy:

1.  **ResNet50 (Primary Backbone)**: A 50-layer Residual Network pretrained on ImageNet. Modified to accept single-channel Grayscale X-rays.
2.  **ResNet18 (Comparison)**: A lighter 18-layer comparison model (RQ1).
3.  **Ensemble Model (RQ3)**: A Voting Ensemble averaging predictions from ResNet50 and ResNet18.

### Preprocessing Pipeline (RQ2)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization to enhance bone structure.
- **Gaussian Blur**: Noise reduction (kernel=5x5).
- **Resize**: Standardized to 224x224.

---

## 🧪 Experiments & Research Questions
This submission answers 5 key Research Questions:
- **RQ1**: ResNet50 vs ResNet18 Performance.
- **RQ2**: Impact of CLAHE+Blur Preprocessing.
- **RQ3**: Efficacy of Ensemble Learning.
- **RQ4**: Correlation between Model Confidence and Rule-Based Severity.
- **RQ5**: Impact of Data Augmentation.

---

## 🚀 How to Reproduce Results (Submission)

### 1. Setup Environment
```bash
pip install torch torchvision opencv-python numpy scikit-image pillow matplotlib
```

### 2. Prepare Data
Ensure your dataset is organized as:
```
data/
├── train/
│   ├── fractured/
│   └── normal/
├── val/
│   ├── fractured/
│   └── normal/
```

### 3. Run the Submission Notebook
The core experiments are contained in `submission_notebook.ipynb`.
1.  Open the notebook:
    ```bash
    jupyter notebook submission_notebook.ipynb
    ```
2.  Run all cells.
3.  **Output**:
    - Trained models saved as `model_r50.pth`, `model_r18.pth`.
    - Plots and Tables generated in `Figures_Tables/RQx/`.

### 4. Figures & Tables Submission
After running the notebook, the `Figures_Tables` directory will contain the required structure for the ZIP submission:
```
Figures_Tables/
├── RQ1/
│   ├── RQ1_Fig1.pdf
├── RQ2/
│   ├── RQ2_Fig1.pdf
...
```

---

## 📂 Project Structure
```
Project/
├── Figures_Tables/       <-- Generated outputs for submission
├── experiments_output/   <-- Intermediate logs
├── notebooks/            <-- Colab & Submission notebooks
├── src/                  <-- Source Code
│   ├── models.py         <-- ResNet50, ResNet18, Ensemble definitions
│   ├── metalearner.py    <-- Stacking Meta-Learner (Logistic Regression)
│   ├── train.py          <-- Training loop & Experiment logic
│   ├── main.py           <-- Inference & Hybrid System logic
│   ├── rq1_backbone.py   <-- RQ1 Experiment Script
│   ├── rq2_preprocessing.py <-- RQ2 Experiment Script
│   ├── rq3_ensemble.py   <-- RQ3 Experiment Script
│   ├── rq4_rule_engine.py <-- RQ4 Experiment Script
│   └── rq5_augmentation.py <-- RQ5 Experiment Script
├── submission_notebook.ipynb <-- MAIN ENTRY POINT
└── README.md             <-- This file
```
