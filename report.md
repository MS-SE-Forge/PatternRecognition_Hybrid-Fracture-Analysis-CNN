# Hybrid Fracture Analysis System - Project Report
> **Course**: Pattern Recognition (Part 2) | **Group**: [Group #]

## 1. Introduction
This project implements a hybrid approach to bone fracture analysis, combining deep convolutional neural networks (CNNs) for detection with deterministic morphological analysis for severity estimation.

### Research Questions
To validate our system, we defined the following five research questions:
1.  **RQ1**: Does a deeper ResNet50 backbone outperform a lighter ResNet18 backbone?
2.  **RQ2**: Does domain-specific preprocessing (CLAHE+Blur) improve classification accuracy?
3.  **RQ3**: Can an ensemble of models improve reliability over a single best model?
4.  **RQ4**: Is there a correlation between model confidence and rule-based severity metrics?
5.  **RQ5**: Does geometric data augmentation improve generalization on this dataset?

---

## 2. Methodology

### 2.1 Dataset & Preprocessing
- **Source**: [Provide Dataset Source]
- **Preprocessing**: Grayscale conversion, CLAHE (ClipLimit=2.0), Gaussian Blur (5x5), Resize (224x224).
- **Augmentation (RQ5)**: Random horizontal flips and rotations (+/- 15 degrees).

### 2.2 Model Architectures
- **Base Learner**: ResNet50, modified for 1-channel input and 2-class output.
- **Comparison Model**: ResNet18 (same modifications).
- **Meta-Learner**: Voting Ensemble averaging logits from Model A and Model B.

### 2.3 Rule-Based Engine
- **Fracture Severity**: Determined by measuring the gap between edge segments using Probabilistic Hough Transform.
- **Rules**:
    - Gap > 2mm: Severe
    - Gap < 1mm & Low Contrast: Hairline
    - Else: Simple Displaced

---

## 3. Results & Discussion

### RQ1: ResNet50 vs ResNet18
*Placeholder for results from `Figures_Tables/RQ1/`. Discuss which model converged faster and achieved higher peak accuracy.*

### RQ2: Preprocessing Efficacy
*Placeholder for results from `Figures_Tables/RQ2/`. Discuss if the "clean" images helped the model or if the raw features were sufficient.*

### RQ3: Ensemble Performance
*Placeholder for results from `Figures_Tables/RQ3/`. Discuss if the ensemble provided a stability boost or higher accuracy.*

### RQ4: Hybrid Analysis
*Placeholder for results from `Figures_Tables/RQ4/`. Analyzes the distribution of "Severe" vs "Hairline" classifications on the validation set.*

### RQ5: Data Augmentation
*Placeholder for results from `Figures_Tables/RQ5/`. Discuss if augmentation reduced overfitting.*

---

## 4. Conclusion
[Summarize main findings. Example: The system demonstrated that ResNet50 with CLAHE preprocessing yields the highest accuracy, while the Rule-Based engine successfully categorized severity in 85% of detected fractures.]

---


## Appendix: Implementation Details
- **Framework**: PyTorch
- **Hardware**: [GPU/CPU used]
- **Submission**: See `submission_notebook.ipynb` and `src/rq*.py` scripts for full experimental code.

