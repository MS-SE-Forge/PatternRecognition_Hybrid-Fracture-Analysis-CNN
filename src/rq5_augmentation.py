#!/usr/bin/env python3
"""
RQ5: Data Augmentation Impact (ResNet50 with vs without augmentation)
Exports:
- Figures_Tables/RQ5/RQ5_Fig1.pdf
- Figures_Tables/RQ5/RQ5_Tab1.xlsx
"""
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from train import train_model_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="./Figures_Tables")
    args = ap.parse_args()

    rq_dir = os.path.join(args.out, "RQ5")
    os.makedirs(rq_dir, exist_ok=True)

    print("Running RQ5 - WITH augmentation...")
    hist_aug, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        use_augmentation=True,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )

    print("Running RQ5 - NO augmentation...")
    hist_no, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        use_augmentation=False,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )

    fig = plt.figure(figsize=(10,6))
    plt.plot(hist_aug["val_acc"], label="With Augmentation")
    plt.plot(hist_no["val_acc"], label="No Augmentation", linestyle="--")
    plt.title("RQ5: Impact of Data Augmentation")
    plt.xlabel("Epochs"); plt.ylabel("Validation Accuracy"); plt.legend(); plt.grid(True)
    fig_path = os.path.join(rq_dir, "RQ5_Fig1.pdf")
    plt.tight_layout(); plt.savefig(fig_path); plt.close(fig)
    print("Saved:", fig_path)

    df = pd.DataFrame([
        {"Setting":"With augmentation", "Final_Val_Acc": float(hist_aug["val_acc"][-1]), "Final_Val_Loss": float(hist_aug["val_loss"][-1])},
        {"Setting":"No augmentation", "Final_Val_Acc": float(hist_no["val_acc"][-1]), "Final_Val_Loss": float(hist_no["val_loss"][-1])},
    ])
    tab_path = os.path.join(rq_dir, "RQ5_Tab1.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Summary")
    print("Saved:", tab_path)

if __name__ == "__main__":
    main()
