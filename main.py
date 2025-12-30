import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

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

        return smoothed_img

# ---------------------------------------------------------
# 2. Deep Learning Module (CNN) [cite: 17, 38]
# ---------------------------------------------------------
class FractureCNN(nn.Module):
    def __init__(self):
        super(FractureCNN, self).__init__()
        # Using ResNet50 as the backbone as per comparison table [cite: 100]
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify first layer to accept grayscale (1 channel) instead of RGB (3)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fully connected layer for binary classification 
        # (Fracture vs No Fracture) [cite: 26]
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 2) 

    def forward(self, x):
        return self.backbone(x)

# ---------------------------------------------------------
# 3. Morphological & Texture Descriptor Module [cite: 28, 76]
# ---------------------------------------------------------
class MorphologicalAnalyzer:
    def analyze_texture(self, image):
        """
        Calculates GLCM features and Edge intensity.
        """
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
        Estimates fracture gap size (displacement) using Hough Transform or contour analysis.
        Simplified here to return a mock pixel gap for demonstration. [cite: 77]
        """
        # In a real scenario, this would calculate the distance between dominant edge lines.
        # Returning a dummy value to demonstrate the logic flow.
        estimated_pixel_gap = 5.0  # e.g., 5 pixels
        return estimated_pixel_gap

# ---------------------------------------------------------
# 4. Hybrid Logic & Rule-Based Classification [cite: 29, 79]
# ---------------------------------------------------------
class HybridSystem:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.cnn = FractureCNN()
        self.morph_analyzer = MorphologicalAnalyzer()
        self.cnn.eval() # Set to evaluation mode

    def analyze_image(self, image_path):
        # Step A: Preprocessing
        processed_img = self.preprocessor.preprocess(image_path)
        
        # Prepare for CNN (Add batch and channel dims)
        img_tensor = transforms.ToTensor()(processed_img).unsqueeze(0)
        
        # Step B: CNN Initial Localization/Detection [cite: 28]
        with torch.no_grad():
            outputs = self.cnn(img_tensor)
            _, predicted = torch.max(outputs, 1)
            cnn_prediction = "Fracture" if predicted.item() == 1 else "Normal"

        if cnn_prediction == "Normal":
            return "Diagnosis: Healthy Bone Structure"

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
    system = HybridSystem()
    
    print("System Initialized. Waiting for X-ray input...")
    # To run this, you would provide a path to an actual image:
    # result = system.analyze_image("path_to_xray.jpg")
    # print(result)