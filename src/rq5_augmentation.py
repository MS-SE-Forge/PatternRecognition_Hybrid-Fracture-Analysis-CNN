#!/usr/bin/env python3
"""
RQ5: Data Augmentation + Meta-Learner (Stacking)

Exports into Figures_Tables/RQ5:
- RQ5_Fig1.pdf  (augmentation comparison)
- RQ5_Tab1.xlsx (augmentation summary)
- RQ5_Fig2.pdf  (meta-learner comparison)  [if --run_meta]
- RQ5_Tab2.xlsx (meta-learner metrics)     [if --run_meta]
"""

import os, argparse, subprocess
import pandas as pd
import matplotlib.pyplot as plt
import sys

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
    ap.add_argument("--r50", default=os.path.join(REPO_ROOT, "model_r50.pth"))
    ap.add_argument("--r18", default=os.path.join(REPO_ROOT, "model_r18.pth"))
    ap.add_argument("--run_meta", action="store_true")
    args = ap.parse_args()

    rq_dir = os.path.join(args.out, "RQ5")
    os.makedirs(rq_dir, exist_ok=True)

    print("Running RQ5-A: WITH augmentation...")
    hist_aug, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        use_augmentation=True,
        use_preprocessing=True,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=os.path.join(rq_dir, "rq5_r50_aug.pth"),
        classmap_path=os.path.join(REPO_ROOT, "class_mapping.json"),
    )

    print("Running RQ5-A: NO augmentation...")
    hist_no, _ = train_model_experiment(
        args.data,
        backbone_name="resnet50",
        use_augmentation=False,
        use_preprocessing=True,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=os.path.join(rq_dir, "rq5_r50_noaug.pth"),
        classmap_path=os.path.join(REPO_ROOT, "class_mapping.json"),
    )

    fig = plt.figure(figsize=(10, 6))
    plt.plot(hist_aug["val_acc"], label="With Augmentation")
    plt.plot(hist_no["val_acc"], label="No Augmentation", linestyle="--")
    plt.title("RQ5: Impact of Data Augmentation")
    plt.xlabel("Epochs"); plt.ylabel("Validation Accuracy")
    plt.legend(); plt.grid(True)

    fig_path = os.path.join(rq_dir, "RQ5_Fig1.pdf")
    plt.tight_layout(); plt.savefig(fig_path); plt.close(fig)
    print("Saved:", fig_path)

    df = pd.DataFrame([
        {"Setting":"With augmentation", "Final_Val_Acc": float(hist_aug["val_acc"][-1]), "Final_Val_Loss": float(hist_aug["val_loss"][-1])},
        {"Setting":"No augmentation", "Final_Val_Acc": float(hist_no["val_acc"][-1]), "Final_Val_Loss": float(hist_no["val_loss"][-1])},
    ])

    tab_path = os.path.join(rq_dir, "RQ5_Tab1.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Augmentation_Summary")
    print("Saved:", tab_path)

    # Optional meta-learner
    if args.run_meta:
        if not (os.path.exists(args.r50) and os.path.exists(args.r18)):
            print("Skipping meta-learner: missing base weights.")
            print("Run RQ1 first to create:")
            print(" -", args.r50)
            print(" -", args.r18)
            return

        meta_script = os.path.join(SRC_DIR, "meta_learner.py")
        if not os.path.exists(meta_script):
            raise FileNotFoundError(f"Missing meta learner script at: {meta_script}")

        print("Running RQ5-B: Meta-learner stacking...")
        subprocess.check_call([
            "python", meta_script,
            "--data", args.data,
            "--batch_size", str(args.batch_size),
            "--r50", args.r50,
            "--r18", args.r18,
            "--out", rq_dir,
            "--prefix", "RQ5"
        ])

if __name__ == "__main__":
    main()
