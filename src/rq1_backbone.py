#!/usr/bin/env python3
"""
RQ1: Backbone Comparison (ResNet50 vs ResNet18)
Trains both backbones under identical conditions and saves:
- Figures_Tables/RQ1/RQ1_Fig1.pdf
- Figures_Tables/RQ1/RQ1_Tab1.xlsx (final metrics summary)
"""
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Import project code
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from train import train_model_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="./Figures_Tables")
    ap.add_argument("--save_r50", default="model_r50.pth")
    ap.add_argument("--save_r18", default="model_r18.pth")
    args = ap.parse_args()

    rq_dir = os.path.join(args.out, "RQ1")
    os.makedirs(rq_dir, exist_ok=True)

    print("Running RQ1 - ResNet50...")
    hist_r50, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_r50
    )

    print("Running RQ1 - ResNet18...")
    hist_r18, _ = train_model_experiment(
        args.data,
        backbone_name="resnet18",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_r18
    )

    if not hist_r50 or not hist_r18:
        raise RuntimeError("Training history missing. Check train_model_experiment returns history dicts.")

    # Plot comparison
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_r50["val_acc"], label="ResNet50 Val Acc", marker="o")
    plt.plot(hist_r18["val_acc"], label="ResNet18 Val Acc", marker="s")
    plt.title("RQ1: Validation Accuracy Comparison")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist_r50["val_loss"], label="ResNet50 Val Loss", marker="o")
    plt.plot(hist_r18["val_loss"], label="ResNet18 Val Loss", marker="s")
    plt.title("RQ1: Validation Loss Comparison")
    plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    fig_path = os.path.join(rq_dir, "RQ1_Fig1.pdf")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    print("Saved:", fig_path)

    # Table (final epoch metrics)
    df = pd.DataFrame([
        {"Model":"ResNet50", "Final_Val_Acc": float(hist_r50["val_acc"][-1]), "Final_Val_Loss": float(hist_r50["val_loss"][-1])},
        {"Model":"ResNet18", "Final_Val_Acc": float(hist_r18["val_acc"][-1]), "Final_Val_Loss": float(hist_r18["val_loss"][-1])},
    ])
    tab_path = os.path.join(rq_dir, "RQ1_Tab1.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Summary")
    print("Saved:", tab_path)

if __name__ == "__main__":
    main()
