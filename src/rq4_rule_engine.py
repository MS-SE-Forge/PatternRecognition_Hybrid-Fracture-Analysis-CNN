#!/usr/bin/env python3
"""
RQ4: Rule Engine Analysis
Runs HybridSystem on a sample of validation fractured images and exports:
- Figures_Tables/RQ4/RQ4_Fig1.pdf (displacement histogram)
- Figures_Tables/RQ4/RQ4_Tab1.xlsx (severity counts + basic stats)
"""
import os, argparse, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from main import HybridSystem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./Figures_Tables")
    ap.add_argument("--model", default="model_r50.pth")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    rq_dir = os.path.join(args.out, "RQ4")
    os.makedirs(rq_dir, exist_ok=True)

    val_frac = glob.glob(os.path.join(args.data, "val", "fractured", "*.*"))
    if not val_frac:
        val_frac = glob.glob(os.path.join(args.data, "validation", "fractured", "*.*"))

    if not val_frac:
        raise FileNotFoundError("No validation fractured images found under data/val/fractured or data/validation/fractured")

    batch = val_frac[:min(args.limit, len(val_frac))]

    system = HybridSystem(model_path=args.model)

    severities, displacements, confidences = [], [], []
    for img_path in batch:
        try:
            res = system.analyze_image(img_path)
            # handle different keys robustly
            primary = (res.get("Primary_Diagnosis") or res.get("prediction") or "").lower()
            if "fract" in primary:
                sev = res.get("Severity") or res.get("severity") or res.get("severity_level") or "Unknown"
                severities.append(sev)
                metrics = res.get("Metrics", {}) or {}
                disp = metrics.get("Displacement_mm", np.nan)
                displacements.append(disp)
                conf = res.get("Confidence", res.get("confidence", np.nan))
                confidences.append(conf)
        except Exception:
            continue

    if len(displacements) == 0:
        raise RuntimeError("No fractures detected/passed rule engine in the sample. Increase --limit or verify HybridSystem outputs.")

    # Figure
    fig = plt.figure(figsize=(10,5))
    plt.hist([d for d in displacements if d==d], bins=10, alpha=0.75)
    plt.title("RQ4: Distribution of Detected Fracture Displacements")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Count")
    plt.grid(True)
    fig_path = os.path.join(rq_dir, "RQ4_Fig1.pdf")
    plt.tight_layout(); plt.savefig(fig_path); plt.close(fig)
    print("Saved:", fig_path)

    # Table
    sev_counts = pd.Series(severities).value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]

    stats = pd.DataFrame([{
        "N_analyzed": len(batch),
        "N_fracture_detected": len(severities),
        "Disp_mean": float(np.nanmean(displacements)),
        "Disp_std": float(np.nanstd(displacements)),
        "Conf_mean": float(np.nanmean(confidences)) if len(confidences) else np.nan
    }])

    tab_path = os.path.join(rq_dir, "RQ4_Tab1.xlsx")
    with pd.ExcelWriter(tab_path, engine="openpyxl") as w:
        sev_counts.to_excel(w, index=False, sheet_name="Severity_Counts")
        stats.to_excel(w, index=False, sheet_name="Summary_Stats")
    print("Saved:", tab_path)

if __name__ == "__main__":
    main()
