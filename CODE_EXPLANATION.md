# 🔍 Codebase Walkthrough

This document explains exactly how the code works, breaking down `src/main.py` (The Analyzer) and `src/train.py` (The Teacher).

---

## 1. `src/main.py`: The "Brain" & Analysis Logic
This file is responsible for taking an image and deciding if it has a fracture.

### 🖼️ Part 1: Preprocessing (`ImagePreprocessor`)
Before the AI looks at an image, we "clean" it to make bones easier to see.

```python
class ImagePreprocessor:
    def __init__(self):
        # We start by getting a "CLAHE" tool ready.
        # CLAHE stands for "Contrast Limited Adaptive Histogram Equalization".
        # In simple terms: It makes dark bones brighter and bright artifacts dimmer so details pop.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, image_path):
        # 1. Read the image in Grayscale (0 means black & white mode)
        img = cv2.imread(image_path, 0)
        
        # 2. Apply the Contrast Enhancer (CLAHE) we prepared earlier
        enhanced_img = self.clahe.apply(img)

        # 3. Apply Gaussian Smoothing (Blurring)
        # This removes "noise" (graininess) which might confuse the AI.
        # (5, 5) is the size of the blurring brush.
        smoothed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        
        # 4. Resize to 224x224 pixels
        # The AI (ResNet) expects a square image of this exact size.
        resized_img = cv2.resize(smoothed_img, (224, 224))

        # 5. Convert to PIL format
        # This is just a file format conversion to keep things consistent for PyTorch.
        return Image.fromarray(resized_img)
```

### 🧠 Part 2: The AI Model (`FractureCNN`)
This is the neural network structure.

```python
class FractureCNN(nn.Module):
    def __init__(self):
        # We download a pre-built brain called "ResNet50" that already knows how to see shapes.
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # MODIFY 1: The original ResNet expects Color (3 channels: Red, Green, Blue).
        # X-rays are Black & White (1 channel). So we change the first "eye" (conv1) to accept 1 channel.
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # MODIFY 2: The original ResNet classifies 1000 things (dogs, cats, cars...).
        # We only want 2 things: "Fractured" or "Normal".
        # So we replace the final "decision maker" (fc) with a layer that outputs 2 numbers.
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 2) 

    def forward(self, x):
        # This function defines how data flows through the brain.
        # It's simple: Image In -> Backbone -> Prediction Out.
        return self.backbone(x)
```

### 📐 Part 3: The Measurement Tool (`MorphologicalAnalyzer`)
If the AI thinks there is a fracture, this math-based tool tries to measure it.

```python
class MorphologicalAnalyzer:
    def measure_displacement(self, edges_map):
        # 1. Find straight lines in the image using "Hough Transform".
        # This looks for straight edges (which broken bones often have).
        lines = cv2.HoughLinesP(edges_map, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        
        # 2. Calculate the gap
        # It looks at all the lines found and calculates the distance between them.
        # The largest distance between parallel bone lines is estimated as the "displacement".
        # (Uses Euclidean distance math: sqrt((x2-x1)^2 + (y2-y1)^2))
        return max_gap
```

---

## 2. `src/train.py`: The Teacher
This script teaches the `FractureCNN` using your data.

### 🔄 The Training Loop (`train_model`)
This is the core function where learning happens.

```python
def train_model(data_dir, ...):
    # 1. Choose Hardware: Checks if you have a GPU (CUDA or MPS) or just a CPU.
    
    # 2. Setup Data Transformers
    # This aligns with what we did in main.py (Grayscale -> CLAHE -> Blur -> Resize).
    # We add `ToTensor()` to turn the image into numbers (0s and 1s) for the AI.
    data_transforms = transforms.Compose([...])

    # 3. Load Data
    # It reads your folders ('train', 'val') and automatically labels images based on folder names.
    train_dataset = datasets.ImageFolder(..., transform=data_transforms)
    
    # 4. START LEARNING (The Loop)
    for epoch in range(num_epochs):
        # Phase 1: Training (Study)
        # It guesses on an image -> Checks if wrong -> Adjusts internal brain weights.
        optimizer.step() 
        
        # Phase 2: Validation (Quiz)
        # It checks its knowledge on the 'val' folder without learning (just testing).
        
        # 5. Save the Best Version
        # If the 'Quiz' score is the highest ever, it saves the brain to 'fracture_model_best.pth'.

    # 6. Final Exam
    # After all epochs, it runs one last test on the 'test' folder to verify true performance.
```

### 🛠️ Key Fixes We Made
You might see `ImageFile.LOAD_TRUNCATED_IMAGES = True`. This tells Python: "If an image file ends abruptly (corrupted download), don't crash. Just read what you can."

---

## Summary of the Flow
1.  **Input**: X-ray Image.
2.  **`ImagePreprocessor`**: Cleans image, makes it 224x224 grayscale.
3.  **`FractureCNN`**: Sees the pattern. output: "Fractured (99%)".
4.  **`MorphologicalAnalyzer`**: If broken, measures the gap (e.g., "2.5mm").
5.  **`HybridSystem`**: Combines both. "High Probability of Fracture with 2.5mm displacement -> SEVERE".
