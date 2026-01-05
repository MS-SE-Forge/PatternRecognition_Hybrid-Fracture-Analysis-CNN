#!/usr/bin/env python3
"""
RQ3: Ensemble Learning (ResNet50 + ResNet18 soft voting)
Requires weights from RQ1:
- model_r50.pth
- model_r18.pth

Exports:
- Figures_Tables/RQ3/RQ3_Fig1.pdf
- Figures_Tables/RQ3/RQ3_Tab1.xlsx (val metrics)
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from models import FractureCNN, EnsembleModel
from main import ImagePreprocessor

import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class OpenCVPreprocessing:
    def __init__(self):
        self.pre = ImagePreprocessor()
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        enhanced = self.pre.clahe.apply(img_np)
        smoothed = cv2.GaussianBlur(enhanced, (5,5), 0)
        resized = cv2.resize(smoothed, (224,224))
        return Image.fromarray(resized)

def eval_model_probs(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
    return np.vstack(probs), np.concatenate(ys)

def metrics_from_probs(probs, y):
    pred = probs.argmax(1)
    acc = (pred == y).mean()
    # binary P/R/F1 on class 1 as positive
    tp = ((pred==1) & (y==1)).sum()
    fp = ((pred==1) & (y==0)).sum()
    fn = ((pred==0) & (y==1)).sum()
    prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    return {"Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="./Figures_Tables")
    ap.add_argument("--r50", default="model_r50.pth")
    ap.add_argument("--r18", default="model_r18.pth")
    args = ap.parse_args()

    rq_dir = os.path.join(args.out, "RQ3")
    os.makedirs(rq_dir, exist_ok=True)

    val_dir = os.path.join(args.data, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(args.data, "validation")

    tf = transforms.Compose([OpenCVPreprocessing(), transforms.ToTensor()])
    val_ds = datasets.ImageFolder(val_dir, transform=tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    r50 = FractureCNN(backbone_name="resnet50")
    r18 = FractureCNN(backbone_name="resnet18")

    if not os.path.exists(args.r50) or not os.path.exists(args.r18):
        raise FileNotFoundError("Missing model weights. Run RQ1 first to create model_r50.pth and model_r18.pth.")

    r50.load_state_dict(torch.load(args.r50, map_location=r50.device))
    r18.load_state_dict(torch.load(args.r18, map_location=r18.device))

    ensemble = EnsembleModel(r50, r18).to(r50.device)

    p50, y = eval_model_probs(r50, val_loader, r50.device)
    p18, _ = eval_model_probs(r18, val_loader, r18.device)
    pens, _ = eval_model_probs(ensemble, val_loader, r50.device)

    m50 = metrics_from_probs(p50, y)
    m18 = metrics_from_probs(p18, y)
    mens = metrics_from_probs(pens, y)

    df = pd.DataFrame([
        {"Model":"ResNet50", **m50},
        {"Model":"ResNet18", **m18},
        {"Model":"Ensemble (avg logits)", **mens},
    ])
    tab_path = os.path.join(rq_dir, "RQ3_Tab1.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Val_Metrics")
    print("Saved:", tab_path)

    fig = plt.figure(figsize=(8,6))
    plt.bar(df["Model"], df["Accuracy"])
    plt.ylim(0, 1.0)
    plt.ylabel("Validation Accuracy")
    plt.title("RQ3: Base Models vs Ensemble")
    plt.xticks(rotation=15, ha="right")
    fig_path = os.path.join(rq_dir, "RQ3_Fig1.pdf")
    plt.tight_layout(); plt.savefig(fig_path); plt.close(fig)
    print("Saved:", fig_path)

if __name__ == "__main__":
    main()
