#!/usr/bin/env python3
"""
meta_learner.py 

Trains a true meta-learner (Logistic Regression) on top of TWO base learners:
- FractureCNN(backbone=resnet50)
- FractureCNN(backbone=resnet18)
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import matplotlib.pyplot as plt

from models import FractureCNN
from train import get_transforms


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"Accuracy": float(acc), "Precision": float(p), "Recall": float(r), "F1": float(f1)}


def predict_proba(model, loader, device):
    model.eval()
    probs_list, y_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_list.append(probs)
            y_list.append(y.numpy())
    return np.vstack(probs_list), np.concatenate(y_list)


def ensure_split_dir(data_dir):
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    return val_dir, test_dir if os.path.exists(test_dir) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--r50", required=True)
    ap.add_argument("--r18", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="RQ5")
    ap.add_argument("--use_test", action="store_true")
    ap.add_argument("--no_preproc", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    use_pre = not args.no_preproc
    eval_tf = get_transforms(augment=False, use_preprocessing=use_pre)

    val_dir, test_dir = ensure_split_dir(args.data)
    eval_dir = (test_dir if (args.use_test and test_dir is not None) else val_dir)
    split_name = "test" if (args.use_test and test_dir is not None) else "val"
    print(f"Meta-learner using split: {split_name} ({eval_dir})")

    eval_ds = datasets.ImageFolder(eval_dir, transform=eval_tf)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if not os.path.exists(args.r50):
        raise FileNotFoundError(f"Missing ResNet50 weights: {args.r50}")
    if not os.path.exists(args.r18):
        raise FileNotFoundError(f"Missing ResNet18 weights: {args.r18}")

    m50 = FractureCNN(backbone_name="resnet50", pretrained=False).to(device)
    m18 = FractureCNN(backbone_name="resnet18", pretrained=False).to(device)

    m50.load_state_dict(torch.load(args.r50, map_location=device))
    m18.load_state_dict(torch.load(args.r18, map_location=device))

    p50, y = predict_proba(m50, eval_loader, device)
    p18, _ = predict_proba(m18, eval_loader, device)

    X = np.column_stack([p50[:, 1], p18[:, 1]]).astype(np.float32)

    meta = LogisticRegression(max_iter=200)
    meta.fit(X, y)

    y_pred50 = p50.argmax(1)
    y_pred18 = p18.argmax(1)
    p_avg = (p50 + p18) / 2.0
    y_pred_avg = p_avg.argmax(1)
    y_pred_meta = meta.predict(X)

    m_res50 = compute_metrics(y, y_pred50)
    m_res18 = compute_metrics(y, y_pred18)
    m_avg = compute_metrics(y, y_pred_avg)
    m_meta = compute_metrics(y, y_pred_meta)

    df = pd.DataFrame([
        {"Model": "ResNet50", **m_res50},
        {"Model": "ResNet18", **m_res18},
        {"Model": "Avg Ensemble (soft voting)", **m_avg},
        {"Model": "Meta-Learner (stacking)", **m_meta},
    ])

    meta_path = os.path.join(args.out, "meta_learner.pkl")
    joblib.dump(meta, meta_path)
    print("Saved meta model:", meta_path)

    tab_path = os.path.join(args.out, f"{args.prefix}_Tab2.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Metrics")
    print("Saved table:", tab_path)

    fig = plt.figure()
    x = np.arange(len(df["Model"]))
    w = 0.2
    plt.bar(x - 1.5*w, df["Accuracy"], w, label="Accuracy")
    plt.bar(x - 0.5*w, df["Precision"], w, label="Precision")
    plt.bar(x + 0.5*w, df["Recall"], w, label="Recall")
    plt.bar(x + 1.5*w, df["F1"], w, label="F1")
    plt.xticks(x, df["Model"], rotation=15, ha="right")
    plt.ylabel("Metric")
    plt.title(f"{args.prefix}: Meta-Learner vs Base/Ensemble ({split_name})")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(args.out, f"{args.prefix}_Fig2.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", fig_path)

    print("\nSummary:\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
