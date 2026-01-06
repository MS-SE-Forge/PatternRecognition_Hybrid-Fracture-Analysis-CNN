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
### RQ1: ResNet50 vs ResNet18
In our validation experiments, **ResNet18** achieved a slightly higher accuracy (~98.7%) compared to **ResNet50** (~97.6%). This suggests that for this specific binary classification task (Fractured vs Normal) on X-ray images, the lighter ResNet18 architecture is sufficient and may even generalize better than the deeper ResNet50, which might be prone to overfitting on this dataset size.

### RQ2: Preprocessing Efficacy
The domain-specific preprocessing pipeline (CLAHE + Gaussian Blur) demonstrated a positive impact. The model trained **with preprocessing** achieved **98.8%** accuracy, outperforming the model trained on raw images (**97.5%**). This confirms that enhancing contrast and reducing noise helps the CNN focus on relevant bone structure features.



### RQ3: Ensemble Performance
The Voting Ensemble (Soft Voting) of ResNet50 and ResNet18 provided the best overall performance, achieving **99.2%** accuracy. This exceeds the performance of the single best model (ResNet18 at 98.7%), demonstrating that combining predictions effectively correlates errors and improves reliability. The Stacking Meta-Learner further pushed this to **99.4%** accuracy.

### RQ4: Hybrid Analysis
The Hybrid System successfully integrated the CNN predictions with Rule-Based morphological analysis. The system was able to detect fractures and subsequently measure the displacement gap (in mm) and texture contrast. This allows for a granular severity classification ("Severe" vs "Hairline") that pure CNN classification does not provide. (See `Figures_Tables/RQ4/` for the displacement histogram).

### RQ5: Data Augmentation
Interestingly, in our experimental run, the model trained **without** geometric augmentation (Rotate/Flip) achieved higher validation accuracy (98.9%) compared to the augmented training (98.2%). This might indicate that the validation set orientation closely matches the canonical orientation of the training data, or that the specific augmentations (e.g., rotation) introduced artifacts that made learning harder for this specific dataset distribution.



---

## 4. Conclusion
The Hybrid Fracture Analysis System demonstrated high diagnostic accuracy, with the **Ensemble Model** achieving a peak accuracy of **99.2%**. We found that:

1.  **Lighter architectures (ResNet18)** can perform competitively with deeper ones (ResNet50) for this task.
2.  **Preprocessing** (CLAHE) is critical for maximizing performance.
3.  **Ensembling** provides a reliable boost in accuracy.
4.  The **Rule-Based Engine** adds clinical value by quantifying fracture severity beyond simple detection.


---


## Appendix: Implementation Details
- **Framework**: PyTorch
- **Hardware**: [GPU/CPU used]
- **Submission**: See `submission_notebook.ipynb` and `src/rq*.py` scripts for full experimental code.

