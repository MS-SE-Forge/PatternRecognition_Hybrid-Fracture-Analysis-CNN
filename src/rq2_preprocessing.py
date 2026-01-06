#!/usr/bin/env python3
"""
RQ2: Preprocessing Impact (CLAHE + Gaussian vs No preprocessing)
Uses ResNet50 as backbone.

Exports:
- Figures_Tables/RQ2/RQ2_Fig1.pdf
- Figures_Tables/RQ2/RQ2_Tab1.xlsx
Also saves a weights file to repo root (optional):
- <repo_root>/model_r50_pre.pth
"""

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Robust repo paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.append(SRC_DIR)

from train import train_model_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "Figures_Tables"))
    ap.add_argument("--save_with_pre", default=os.path.join(REPO_ROOT, "model_r50_pre.pth"))
    args = ap.parse_args()

    rq_dir = os.path.join(args.out, "RQ2")
    os.makedirs(rq_dir, exist_ok=True)

    print("Running RQ2 - WITH preprocessing...")
    hist_with, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        use_preprocessing=True,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_with_pre,
        classmap_path=os.path.join(REPO_ROOT, "class_mapping.json"),
    )

    print("Running RQ2 - NO preprocessing...")
    hist_without, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        use_preprocessing=False,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=None,
        classmap_path=os.path.join(REPO_ROOT, "class_mapping.json"),
    )

    fig = plt.figure(figsize=(10, 6))
    plt.plot(hist_with["val_acc"], label="With Preprocessing (CLAHE+Blur)")
    plt.plot(hist_without["val_acc"], label="No Preprocessing", linestyle="--")
    plt.title("RQ2: Impact of Preprocessing on Accuracy")
    plt.xlabel("Epochs"); plt.ylabel("Validation Accuracy")
    plt.legend(); plt.grid(True)

    fig_path = os.path.join(rq_dir, "RQ2_Fig1.pdf")
    plt.tight_layout(); plt.savefig(fig_path); plt.close(fig)
    print("Saved:", fig_path)

    df = pd.DataFrame([
        {"Setting":"With preprocessing", "Final_Val_Acc": float(hist_with["val_acc"][-1]), "Final_Val_Loss": float(hist_with["val_loss"][-1])},
        {"Setting":"No preprocessing", "Final_Val_Acc": float(hist_without["val_acc"][-1]), "Final_Val_Loss": float(hist_without["val_loss"][-1])},
    ])

    tab_path = os.path.join(rq_dir, "RQ2_Tab1.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Summary")
    print("Saved:", tab_path)

    print("\nSaved preprocessing-trained weights to:")
    print(" -", args.save_with_pre)

if __name__ == "__main__":
    main()
