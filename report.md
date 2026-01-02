# Fracture Analysis Model - Expected Output Report

This document outlines the expected output logs generated during the execution of the Hybrid Fracture Analysis CNN. It breaks down the process into initialization, training, evaluation, and inference phases.

## 1. System Initialization
Upon starting, the system initializes the environment and loads the necessary resources.

- **Device Detection**: The script automatically detects and utilizes the available hardware accelerator.
  - *Expected Log*: `Using device: CUDA (NVIDIA GPU)`
- **Class Detection**: Identifies the target classes for classification.
  - *Expected Log*: `Classes found: ['fractured', 'normal']`
- **Mapping Preservation**: Saves the class-to-index mapping for consistent inference later.
  - *Expected Log*: `Saved class mapping to 'class_mapping.json'`
- **Model Loading**: Downloads the pre-trained ResNet50 weights if not present.
  - *Expected Log*: `Downloading: "https://download.pytorch.org/models/resnet50..."`

## 2. Training Phase
The model undergoes training for a specified number of epochs (e.g., 25).

### Progression Highlights
- **Improvement Tracking**: The system tracks Validation Loss and Accuracy. It saves a "New best model" whenever validation accuracy improves or loss decreases significantly.
- **Accuracy Milestones**:
  - Early epochs (1-5) typically show rapid improvement (e.g., rising from ~83% to ~96%).
  - Later epochs refine the model, stabilizing around high accuracy (>98%).

### Example Log Segment
```text
Epoch 1/25
----------
Train Loss: 0.4058 Acc: 0.8313
Val Loss: 0.5212 Acc: 0.8191
New best model saved!
...
Epoch 25/25
----------
Train Loss: 0.0054 Acc: 0.9982
Val Loss: 0.0371 Acc: 0.9879
New best model saved!
```

- **Objective**: The goal is to maximize `Val Acc` (Validation Accuracy) and minimize `Val Loss`.
- **Result**: In the provided run, the Best Validation Accuracy reached **0.9879**.

## 3. Final Evaluation
After training completes, the model is evaluated against a separate Test Set to ensure generalizability.

- **Metrics**:
  - **Test Set Accuracy**: Indicates the percentage of correctly classified images in the unseen test set (e.g., `0.9921` or 99.21%).
  - **Test Set Loss**: Indicates the error margin (e.g., `0.0315`).

## 4. Inference Execution
The system automatically proceeds to run inference on a designated input directory.

### Setup
- **Directories**:
  - Input: `./data/inference_input` (Folder for user uploads)
  - Output: `./data/inference_results` (Folder for analysis results)
- **Model Reloading**: The best performing model (`fracture_model_best.pth`) is reloaded to ensure the best weights are used.

### Processing Details
The system iterates through images in the input folder and generates predictions.

**Sample Log Output**:
```text
System Initialized. Processing images from '/content/PatternRecognition_Hybrid-Fracture-Analysis-CNN/data/inference_input'...
Analyzing: 1-rotated2-rotated1-rotated1.jpg
Analyzing: tibia_fib_fracture_a1.jpg
...
Processing complete. 7 images analyzed.
```

## Summary
Successful execution is characterized by:
1.  Error-free download and initialization of ResNet50.
2.  Steady decrease in training loss across epochs.
3.  High final test accuracy (>95% typically).
4.  Successful generation of inference results in the designated output folder.
