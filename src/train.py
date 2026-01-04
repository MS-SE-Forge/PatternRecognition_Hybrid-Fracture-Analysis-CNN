
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from models import FractureCNN # Import from our new clean file

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------
# Custom Transform (Modular)
# ---------------------------------------------------------
class OpenCVPreprocessing:
    """
    Applies the inference pipeline preprocessing:
    Grayscale -> CLAHE -> Gaussian Smooth
    """
    def __init__(self, use_clahe=True, use_smoothing=True):
        self.use_clahe = use_clahe
        self.use_smoothing = use_smoothing
        if use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        # img is a PIL Image
        img_np = np.array(img)
        
        # Convert to grayscale if it's RGB
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
        # Apply CLAHE
        if self.use_clahe:
            img_np = self.clahe.apply(img_np)
        
        # Apply Gaussian Smoothing
        if self.use_smoothing:
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
        
        # Resize to 224x224 (ResNet standard)
        img_np = cv2.resize(img_np, (224, 224))
        
        # Return PIL Image (Mode 'L')
        return Image.fromarray(img_np)

def get_transforms(augment=True, use_preprocessing=True):
    """
    Factory for transform pipelines.
    """
    if use_preprocessing:
        preprocess_step = OpenCVPreprocessing(use_clahe=True, use_smoothing=True)
    else:
        # "Raw" means just resize (and maybe grayscale to match channel dims)
        # We use the same class but disable the specific algo steps
        preprocess_step = OpenCVPreprocessing(use_clahe=False, use_smoothing=False)

    if augment:
        return transforms.Compose([
            preprocess_step,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            preprocess_step,
            transforms.ToTensor(),
        ])

def train_model_experiment(
    data_dir, 
    backbone_name='resnet50', 
    use_augmentation=True, 
    use_preprocessing=True,
    num_epochs=15, 
    batch_size=32, 
    learning_rate=0.001,
    save_path=None
):
    """
    Refactored training function that returns history for plotting.
    """
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Transforms
    train_transforms = get_transforms(augment=use_augmentation, use_preprocessing=use_preprocessing)
    val_transforms = get_transforms(augment=False, use_preprocessing=use_preprocessing)

    # Directories
    val_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, 'validation')
    
    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    except FileNotFoundError:
        print(f"Error: Could not find dataset in {data_dir}")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = FractureCNN(backbone_name=backbone_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # History
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_acc = 0.0

    print(f"Starting training: {backbone_name} | Aug: {use_augmentation} | Pre: {use_preprocessing}")

    for epoch in range(num_epochs):
        # Train
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
        
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Val
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

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)

        # Record
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        print(f"Ep {epoch+1}/{num_epochs} - T.Loss: {epoch_loss:.3f} T.Acc: {epoch_acc:.3f} | V.Loss: {val_epoch_loss:.3f} V.Acc: {val_epoch_acc:.3f}")

        # Save Best
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
    
    print(f"Best Val Acc: {best_acc:.4f}")
    if save_path:
        print(f"Model saved to {save_path}")
        
    return history, model

if __name__ == "__main__":
    # Default behavior for command line
    DATASET_PATH = "./data"
    if os.path.exists(DATASET_PATH):
        train_model_experiment(DATASET_PATH, save_path="fracture_model_best.pth")
    else:
        print("Dataset not found at ./data")
