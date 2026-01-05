import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import json
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models import FractureCNN  # from src/models.py

# =========================================================
# Path helpers (FIXES your "double repo folder" issue)
# =========================================================
# This file is: <repo_root>/src/main.py
# So repo_root = parent of src/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "fracture_model_best.pth")
DEFAULT_CLASSMAP_PATH = os.path.join(REPO_ROOT, "class_mapping.json")

DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_INPUT_DIR = os.path.join(DEFAULT_DATA_DIR, "inference_input")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_DIR, "inference_results")


# ---------------------------------------------------------
# 1. Preprocessing Pipeline
# ---------------------------------------------------------
class ImagePreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, image_path):
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

        enhanced_img = self.clahe.apply(img)
        smoothed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        resized_img = cv2.resize(smoothed_img, (224, 224))

        return Image.fromarray(resized_img)  # PIL Image, mode 'L'


# ---------------------------------------------------------
# 2. Morphological & Texture Descriptor Module
# ---------------------------------------------------------
class MorphologicalAnalyzer:
    def analyze_texture(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges) / edges.size

        glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        return {
            "edge_density": edge_density,
            "glcm_contrast": contrast,
            "glcm_homogeneity": homogeneity,
            "edges_map": edges
        }

    def measure_displacement(self, edges_map):
        lines = cv2.HoughLinesP(edges_map, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        if lines is None:
            return 0.0

        max_gap = 0.0
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]

                mid1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                mid2 = np.array([(x3 + x4) / 2, (y3 + y4) / 2])

                gap = np.linalg.norm(mid1 - mid2)
                if gap > max_gap:
                    max_gap = gap

        return max_gap


# ---------------------------------------------------------
# 3. Hybrid Logic & Rule-Based Classification
# ---------------------------------------------------------
class HybridSystem:
    def __init__(self, model_path=None, classmap_path=None):
        self.preprocessor = ImagePreprocessor()
        self.cnn = FractureCNN()
        self.morph_analyzer = MorphologicalAnalyzer()

        # Resolve paths safely
        model_path = model_path or DEFAULT_MODEL_PATH
        classmap_path = classmap_path or DEFAULT_CLASSMAP_PATH

        # Load class mapping (prefer file; fallback to alphabetical assumption)
        self.idx_to_class = None
        if os.path.exists(classmap_path):
            try:
                with open(classmap_path, "r") as f:
                    class_to_idx = json.load(f)
                self.idx_to_class = {v: k for k, v in class_to_idx.items()}
                print(f"Loaded class mapping from: {classmap_path} -> {self.idx_to_class}")
            except Exception as e:
                print(f"Warning: Failed to load class mapping ({classmap_path}): {e}")
        else:
            print(f"Warning: class_mapping.json not found at {classmap_path}. Using default alphabetical assumption.")

        # Load model weights
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location="cpu")
                self.cnn.load_state_dict(state)
                print(f"Loaded weights from: {model_path}")
            except Exception as e:
                print(f"Warning: Error loading weights from {model_path}: {e}")
                print("Using random weights.")
        else:
            print(f"Warning: Model file not found at {model_path}. Using random weights.")

        # Ensure same device
        self.cnn.to(self.cnn.device)
        self.cnn.eval()

    def _cnn_predict_label(self, img_tensor):
        with torch.no_grad():
            outputs = self.cnn(img_tensor)
            _, predicted = torch.max(outputs, 1)
            idx = predicted.item()

        if self.idx_to_class is not None:
            return self.idx_to_class.get(idx, "Unknown")

        # Fallback assumption: alphabetical class order in ImageFolder
        # If your folders are fractured/normal => fractured < normal => idx 0=fractured, 1=normal
        return "fractured" if idx == 0 else "normal"

    def analyze_image(self, image_path):
        processed_img = self.preprocessor.preprocess(image_path)
        img_tensor = transforms.ToTensor()(processed_img).unsqueeze(0).to(self.cnn.device)

        cnn_prediction = self._cnn_predict_label(img_tensor)

        if cnn_prediction.lower() in ["normal", "non-fractured", "healthy"]:
            return {
                "Primary_Diagnosis": "Healthy Bone Structure",
                "Severity": "None",
                "Metrics": {"Displacement_mm": 0.0, "Texture_Contrast": 0.0}
            }

        features = self.morph_analyzer.analyze_texture(processed_img)
        displacement_px = self.morph_analyzer.measure_displacement(features["edges_map"])
        displacement_mm = displacement_px * 0.264  # demo scale

        severity = self.apply_clinical_rules(displacement_mm, features["glcm_contrast"])

        return {
            "Primary_Diagnosis": cnn_prediction,
            "Severity": severity,
            "Metrics": {
                "Displacement_mm": round(displacement_mm, 2),
                "Texture_Contrast": round(features["glcm_contrast"], 2)
            }
        }

    def apply_clinical_rules(self, displacement, contrast):
        if displacement > 2.0:
            return "Severe (Surgical attention required)"
        elif displacement < 1.0 and contrast < 50:
            return "Hairline / Nondisplaced"
        else:
            return "Simple Displaced"


# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Always resolve model from repo root (fixes your warning)
    system = HybridSystem(model_path=DEFAULT_MODEL_PATH, classmap_path=DEFAULT_CLASSMAP_PATH)

    # Always use repo-root data folders (fixes the double-path confusion)
    INPUT_DIR = DEFAULT_INPUT_DIR
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    # Auto-create dirs (so you don’t get "not found" anymore)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nRepo root: {REPO_ROOT}")
    print(f"Processing images from: {INPUT_DIR}")
    print(f"Saving results to:     {OUTPUT_DIR}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for root, _, fs in os.walk(INPUT_DIR):
        for f in fs:
            if os.path.splitext(f)[1].lower() in valid_extensions:
                files.append(os.path.join(root, f))

    if not files:
        print(f"\nNo images found in '{INPUT_DIR}'.")
        print("Add X-ray images there (jpg/png/etc), then run this script again.")
        raise SystemExit(0)

    processed_count = 0
    for image_path in files:
        file = os.path.basename(image_path)
        print(f"\nAnalyzing: {file}")
        try:
            result = system.analyze_image(image_path)

            base_name = os.path.splitext(file)[0]
            output_file = os.path.join(OUTPUT_DIR, f"{base_name}_result.txt")

            with open(output_file, "w") as f:
                f.write(f"Image: {file}\n")
                f.write("-" * 40 + "\n")
                for key, value in result.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_val in value.items():
                            f.write(f"  {sub_key}: {sub_val}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("-" * 40 + "\n")

            processed_count += 1
        except Exception as e:
            print(f"Failed to analyze {file}: {e}")

    print(f"\nDone. {processed_count} images analyzed.")
    print(f"Check '{OUTPUT_DIR}' for result files.")
