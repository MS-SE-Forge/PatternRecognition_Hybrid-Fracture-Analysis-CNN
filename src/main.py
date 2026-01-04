import warnings
# Filter deprecated warnings from torchvision
warnings.filterwarnings("ignore", category=UserWarning) 

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from models import FractureCNN # Import from models.py

# ---------------------------------------------------------
# 1. Preprocessing Pipeline [cite: 16, 70]
# ---------------------------------------------------------
class ImagePreprocessor:
    def __init__(self):
        # CLAHE setup for contrast enhancement 
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, image_path):
        """
        Applies Normalization, CLAHE, and Gaussian Smoothing.
        """
        # Read image in grayscale
        img = cv2.imread(image_path, 0)
        if img is None:
            raise ValueError("Image not found.")

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        enhanced_img = self.clahe.apply(img)

        # Apply Gaussian Smoothing to reduce noise 
        # Using a 5x5 kernel as standard for noise reduction
        smoothed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        
        # Resize to 224x224 (Standard info for ResNet)
        # This ensures all images in a batch are the same size
        resized_img = cv2.resize(smoothed_img, (224, 224))

        # Return PIL Image (Mode 'L') to ensure correct tensor shape (1, H, W)
        return Image.fromarray(resized_img)

# ---------------------------------------------------------
# 2. Deep Learning Module (CNN) [cite: 17, 38]
# ---------------------------------------------------------
# FractureCNN is now imported from models.py to ensure consistency
# between training and inference scripts.

# ---------------------------------------------------------
# 3. Morphological & Texture Descriptor Module [cite: 28, 76]
# ---------------------------------------------------------
class MorphologicalAnalyzer:
    def analyze_texture(self, image):
        """
        Calculates GLCM features and Edge intensity.
        """
        # Convert PIL Image to Numpy Array if needed
        # OpenCV needs numpy arrays, but Preprocessor returns PIL
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # 1. Edge Detection (Sobel/Canny) 
        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges) / edges.size

        # 2. GLCM (Gray-Level Co-occurrence Matrix) 
        # We compute GLCM to find contrast and homogeneity
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
        """
        Estimates fracture gap size (displacement) using Hough Transform.
        Detects line segments in the edge map and calculates the maximum distance
        between parallel-like segments to approximate the fracture gap. [cite: 77]
        """
        # Detect lines using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges_map, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return 0.0

        max_gap = 0.0
        
        # Simple heuristic: Calculate distance between all pairs of lines
        # In a production system, we would filter for parallel lines specifically within the ROI.
        # Here we iterate to find the widest meaningful gap which suggests displacement.
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                
                # Calculate midpoints
                mid1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                mid2 = np.array([(x3 + x4) / 2, (y3 + y4) / 2])
                
                # Euclidean distance between midpoints of two line segments
                gap = np.linalg.norm(mid1 - mid2)
                
                if gap > max_gap:
                    max_gap = gap

        return max_gap

# ---------------------------------------------------------
# 4. Hybrid Logic & Rule-Based Classification [cite: 29, 79]
# ---------------------------------------------------------
class HybridSystem:
    def __init__(self, model_path=None):
        self.preprocessor = ImagePreprocessor()
        self.cnn = FractureCNN()
        self.morph_analyzer = MorphologicalAnalyzer()
        
        # Load class mapping if it exists
        try:
            import json
            with open('class_mapping.json', 'r') as f:
                class_to_idx = json.load(f)
                self.idx_to_class = {v: k for k, v in class_to_idx.items()}
                print(f"Loaded class mapping: {self.idx_to_class}")
        except FileNotFoundError:
            print("Warning: class_mapping.json not found. Using default alphabetical assumption.")
        
        if model_path:
            try:
                self.cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Loaded weights from {model_path}")
            except FileNotFoundError:
                print(f"Warning: Model file {model_path} not found. Using random weights.")
            except Exception as e:
                print(f"Error loading weights: {e}")

        # Ensure model parameters are on the correct device (GPU/MPS/CPU)
        # This fixes the "Input type (cuda) and weight type (cpu) should be the same" error
        self.cnn.to(self.cnn.device)
        self.cnn.eval() # Set to evaluation mode

    def analyze_image(self, image_path):
        # Step A: Preprocessing
        processed_img = self.preprocessor.preprocess(image_path)
        
        # Prepare for CNN (Add batch and channel dims)
        img_tensor = transforms.ToTensor()(processed_img).unsqueeze(0).to(self.cnn.device)
        
        # Step B: CNN Initial Localization/Detection [cite: 28]
        with torch.no_grad():
            outputs = self.cnn(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
            # Determine class label dynamically
            idx = predicted.item()
            if hasattr(self, 'idx_to_class'):
                cnn_prediction = self.idx_to_class.get(idx, "Unknown")
            else:
                # Fallback if no mapping found (assuming alphabetical: fractured=0, normal=1)
                # Standard convention: 'fractured' < 'normal', so 0=fractured, 1=normal
                if idx == 0:
                    cnn_prediction = "fractured"
                else: 
                     cnn_prediction = "normal"

        # Check against "Normal" or synonyms
        if cnn_prediction.lower() in ["normal", "non-fractured", "healthy"]:
            # Return same dictionary structure, but with healthy details
            return {
                "Primary_Diagnosis": "Healthy Bone Structure",
                "Severity": "None",
                "Metrics": {
                    "Displacement_mm": 0.0,
                    "Texture_Contrast": 0.0
                }
            }

        # Step C: Secondary Analysis (Morphology) [cite: 28]
        features = self.morph_analyzer.analyze_texture(processed_img)
        displacement_px = self.morph_analyzer.measure_displacement(features['edges_map'])
        
        # Convert pixels to mm (assuming arbitrary scale factor for demo)
        displacement_mm = displacement_px * 0.264 

        # Step D: Rule-Based Severity Classification [cite: 29, 79]
        severity = self.apply_clinical_rules(displacement_mm, features['glcm_contrast'])

        return {
            "Primary_Diagnosis": cnn_prediction,
            "Severity": severity,
            "Metrics": {
                "Displacement_mm": round(displacement_mm, 2),
                "Texture_Contrast": round(features['glcm_contrast'], 2)
            }
        }

    def apply_clinical_rules(self, displacement, contrast):
        """
        Deterministic logic based on 'Model Design' section[cite: 79, 80, 81].
        """
        # Logic Rule 1: Severe if displacement > 2mm 
        if displacement > 2.0:
            return "Severe (Surgical attention required)"
        
        # Logic Rule 2: Hairline if gap is small and contrast is subtle [cite: 81]
        elif displacement < 1.0 and contrast < 50:
            return "Hairline / Nondisplaced"
        
        # Default Logic
        else:
            return "Simple Displaced"

# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialize the Hybrid System
    # Tries to load the best model if it exists from training
    system = HybridSystem(model_path="fracture_model_best.pth")
    
    # ---------------------------------------------------------
    # Batch Processing Configuration
    # ---------------------------------------------------------
    # Logic to find the data directory
    # Priority 1: 'data' folder in the current working directory (where the user is running the command)
    # This addresses the issue where the user might have a nested structure but is working from the root
    cwd_data_input = os.path.join(os.getcwd(), "data", "inference_input")
    cwd_data_output = os.path.join(os.getcwd(), "data", "inference_results")

    # Priority 2: 'data' folder relative to this script file (in case they are running from a weird location)
    script_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up from src to project root
    script_data_input = os.path.join(script_base_dir, "data", "inference_input")
    script_data_output = os.path.join(script_base_dir, "data", "inference_results")

    if os.path.exists(cwd_data_input):
        INPUT_DIR = cwd_data_input
        OUTPUT_DIR = cwd_data_output
        print(f"Using input directory from current location: {INPUT_DIR}")
    else:
        INPUT_DIR = script_data_input
        OUTPUT_DIR = script_data_output
        print(f"WARNING: 'data/inference_input' not found in current directory. Falling back to script-relative path: {INPUT_DIR}")
    
    print(f"System Initialized. Processing images from '{INPUT_DIR}'...")
    print(f"Results will be saved to '{OUTPUT_DIR}'...")

    if not os.path.exists(INPUT_DIR):
        print(f"Input directory '{INPUT_DIR}' not found. Please create it and add X-ray images.")
    else:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Iterate over all files in the input directory
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        processed_count = 0
        
        for root, dirs, files in os.walk(INPUT_DIR):
            for file in files:
               if os.path.splitext(file)[1].lower() in valid_extensions:
                    image_path = os.path.join(root, file)
                    print(f"Analyzing: {file}")
                    
                    try:
                        # Analyze the image
                        result = system.analyze_image(image_path)
                        
                        # Save result to a text file
                        base_name = os.path.splitext(file)[0]
                        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_result.txt")
                        
                        with open(output_file, "w") as f:
                            f.write(f"Image: {file}\n")
                            f.write("-" * 20 + "\n")
                            for key, value in result.items():
                                if isinstance(value, dict):
                                    f.write(f"{key}:\n")
                                    for sub_key, sub_val in value.items():
                                        f.write(f"  {sub_key}: {sub_val}\n")
                                else:
                                    f.write(f"{key}: {value}\n")
                            f.write("-" * 20 + "\n")
                            
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"Failed to analyze {file}: {e}")

        print(f"\nProcessing complete. {processed_count} images analyzed.")
        print(f"Check '{OUTPUT_DIR}' for detailed result files.")