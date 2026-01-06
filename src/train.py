import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile

from models import FractureCNN  # src/models.py

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================================================
# Robust repo paths (fix Colab / nested working-dir issues)
# =========================================================
# train.py is: <repo_root>/src/train.py  → repo_root = parent of src/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "fracture_model_best.pth")
DEFAULT_CLASSMAP_PATH = os.path.join(REPO_ROOT, "class_mapping.json")


# ---------------------------------------------------------
# Custom Transform (Modular)
# ---------------------------------------------------------
class OpenCVPreprocessing:
    """
    Applies preprocessing:
    Grayscale -> CLAHE -> Gaussian Smooth -> Resize(224,224)
    """
    def __init__(self, use_clahe=True, use_smoothing=True):
        self.use_clahe = use_clahe
        self.use_smoothing = use_smoothing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    def __call__(self, img):
        img_np = np.array(img)

        # Convert to grayscale if RGB
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        if self.use_clahe:
            img_np = self.clahe.apply(img_np)

        if self.use_smoothing:
            img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

        img_np = cv2.resize(img_np, (224, 224))
        return Image.fromarray(img_np)


def get_transforms(augment=True, use_preprocessing=True):
    """
    Factory for transform pipelines.
    """
    if use_preprocessing:
        preprocess_step = OpenCVPreprocessing(use_clahe=True, use_smoothing=True)
    else:
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


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model_experiment(
    data_dir,
    backbone_name="resnet50",
    use_augmentation=True,
    use_preprocessing=True,
    num_epochs=15,
    batch_size=32,
    learning_rate=0.001,
    save_path=None,
    classmap_path=None,
):
    """
    Train model and return (history, model).
    Also saves class mapping JSON if classmap_path is provided.
    """
    device = _get_device()
    print(f"Device: {device}")

    # Transforms
    train_transforms = get_transforms(augment=use_augmentation, use_preprocessing=use_preprocessing)
    val_transforms = get_transforms(augment=False, use_preprocessing=use_preprocessing)

    # Directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset not found. Expected:\n"
            f"  {train_dir}\n"
            f"  {val_dir}\n"
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Save class mapping (IMPORTANT for main.py)
    if classmap_path:
        with open(classmap_path, "w") as f:
            json.dump(train_dataset.class_to_idx, f)
        print(f"Saved class mapping to: {classmap_path}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = FractureCNN(backbone_name=backbone_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    print(f"Starting training: {backbone_name} | Aug={use_augmentation} | Pre={use_preprocessing}")

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset)

        # Val
        model.eval()
        val_loss_sum = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss_sum += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_loss = val_loss_sum / len(val_dataset)
        val_acc = val_corrects / len(val_dataset)

        history["train_loss"].append(float(epoch_loss))
        history["train_acc"].append(float(epoch_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(
            f"Ep {epoch+1}/{num_epochs} "
            f"- T.Loss: {epoch_loss:.3f} T.Acc: {epoch_acc:.3f} "
            f"| V.Loss: {val_loss:.3f} V.Acc: {val_acc:.3f}"
        )

        # Save best
        if save_path and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"Best Val Acc: {best_acc:.4f}")
    if save_path:
        print(f"Best model saved to: {save_path}")

    return history, model


if __name__ == "__main__":
    # Default behavior for CLI
    # Always use repo-root data path + save files to repo root
    if os.path.exists(DEFAULT_DATASET_DIR):
        train_model_experiment(
            data_dir=DEFAULT_DATASET_DIR,
            backbone_name="resnet50",
            use_augmentation=True,
            use_preprocessing=True,
            num_epochs=15,
            batch_size=32,
            learning_rate=0.001,
            save_path=DEFAULT_MODEL_PATH,
            classmap_path=DEFAULT_CLASSMAP_PATH,
        )
    else:
        print(f"Dataset not found at: {DEFAULT_DATASET_DIR}")
        print("Expected repo structure: <repo_root>/data/train and <repo_root>/data/val")
