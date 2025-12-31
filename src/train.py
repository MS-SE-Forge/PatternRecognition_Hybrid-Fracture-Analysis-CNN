
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
from main import FractureCNN, ImagePreprocessor

# ---------------------------------------------------------
# Custom Transform to match Main Pipeline Preprocessing
# ---------------------------------------------------------
class OpenCVPreprocessing:
    """
    Applies the exact same preprocessing as the inference pipeline:
    Grayscale -> CLAHE -> Gaussian Smooth
    """
    def __init__(self):
        self.preprocessor = ImagePreprocessor()

    def __call__(self, img):
        # img is a PIL Image coming from ImageFolder
        # Convert PIL to numpy array (OpenCV format)
        img_np = np.array(img)
        
        # Convert to grayscale if it's RGB
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # Apply CLAHE
        enhanced_img = self.preprocessor.clahe.apply(img_np)
        
        # Apply Gaussian Smoothing
        smoothed_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
        
        # Convert back to PIL Image for ToTensor transform
        return smoothed_img

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Check device: CUDA > MPS (Mac) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU (Slow)")

    # Define transforms
    # We use our custom preprocessing to ensure consistency with inference
    data_transforms = transforms.Compose([
        OpenCVPreprocessing(),
        transforms.ToTensor(), # Converts 0-255 numpy to 0-1 float tensor
        # No normalization/standardization here as main.py doesn't use it explicitly beyond ToTensor()
        # but ResNet usually expects it. For now keeping consistent with main.py structure.
    ])

    # Handle 'val' vs 'validation' folder naming
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, 'validation')
    
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
    except FileNotFoundError:
        print(f"Error: Could not find 'train' or 'val'/'validation' directories in {data_dir}")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Classes found: {train_dataset.classes}")
    # Save class mapping for inference
    import json
    with open('class_mapping.json', 'w') as f:
        json.dump(train_dataset.class_to_idx, f)
    print("Saved class mapping to 'class_mapping.json'")

    # Initialize Model
    model = FractureCNN()
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training Loop
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "fracture_model_best.pth")
            print("New best model saved!")

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")
    print("Model saved to 'fracture_model_best.pth'")

    # ---------------------------------------------------------
    # Final Testing Phase
    # ---------------------------------------------------------
    print("\n" + "="*30)
    print("FINAL EVALUATION ON TEST SET")
    print("="*30)

    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Load the best model we just saved
        model.load_state_dict(torch.load("fracture_model_best.pth"))
        model.eval()
        
        test_loss = 0.0
        test_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                test_corrects += torch.sum(preds == labels.data)
        
        test_loss = test_loss / len(test_dataset)
        test_acc = test_corrects.double() / len(test_dataset)
        
        print(f"Test Set Accuracy: {test_acc:.4f}")
        print(f"Test Set Loss: {test_loss:.4f}")
    else:
        print(f"Warning: 'test' directory not found at {test_dir}. Skipping final evaluation.")

if __name__ == "__main__":
    # Example usage
    # User needs to provide the path to their dataset
    DATASET_PATH = "./data" 
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset directory '{DATASET_PATH}' not found.")
        print("Please create a 'data' folder with 'train' and 'val' subfolders adjacent to this script.")
    else:
        train_model(DATASET_PATH)
