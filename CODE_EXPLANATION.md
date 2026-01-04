# 🔍 Codebase Walkthrough

This document explains the technical implementation of the Hybrid Fracture Analysis System, breaking down the refactored modular structure designed for Part 2 submission.

---

## 1. `src/models.py`: The Neural Architectures
This file defines the Deep Learning models. We moved the model definitions here to allow both training and inference scripts to share the exact same code.

### 🧠 `FractureCNN`
The primary classifier. It can be initialized with different backbones:
- **ResNet50**: The default heavy backbone for maximum accuracy.
- **ResNet18**: A lighter backbone for speed/efficiency comparisons (RQ1).

```python
class FractureCNN(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        # Supports switching between ResNet50 and ResNet18 dynamically
        # Modified conv1 to accept Grayscale (1-channel) input
        ...
```

### 🤝 `EnsembleModel`
A voting ensemble used for RQ3. It takes two trained models and averages their predictions to improve reliability.

---

## 2. `src/main.py`: The Hybrid Inference Engine
This file contains the logic for analyzing new images using the trained models + rule-based logic.

### 🖼️ Part 1: Preprocessing (`ImagePreprocessor`)
Ensures all images look the same before Analysis:
1.  **Grayscale**: Simplifies data.
2.  **CLAHE**: Enhances contrast to make bones pop.
3.  **Gaussian Blur**: Removes noise (grain).
4.  **Resize**: Standardizes to 224x224.

### 📐 Part 2: Rule-Based Logic (`MorphologicalAnalyzer`)
If the CNN detects a fracture, this module measures it:
1.  **Edge Detection**: Finds bone edges.
2.  **Hough Transform**: Finds straight lines.
3.  **Displacement Metric**: Measures the gap between broken pieces to estimate severity (Hairline vs Severe).

---

## 3. `src/train.py`: The Experiment Manager
Refactored to support the Research Question experiments.

### 🧪 `train_model_experiment`
Instead of a hardcoded loop, this function is flexible:
- **`backbone_name`**: Train ResNet50 or ResNet18.
- **`use_preprocessing`**: Toggle CLAHE/Blur on/off (for RQ2).
- **`use_augmentation`**: Toggle geometric flips/rotations (for RQ5).
- **Returns History**: Returns loss/accuracy logs so the Notebook can plot graphs.

---

## 4. `submission_notebook.ipynb`: The Driver
This is the main entry point for the Project Part 2 submission.
- **Runs all Experiments**: Executes RQ1 through RQ5.
- **Generates Figures**: Automatically saves plots to `Figures_Tables/`.
- **Reproducible**: Runs end-to-end to verify the entire pipeline.
