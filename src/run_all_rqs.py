#!/usr/bin/env python3
import argparse, subprocess, zipfile
from pathlib import Path

def zip_figures(out_dir, zip_name="Figures_Tables.zip"):
    out_dir = Path(out_dir)
    zip_path = out_dir.parent / zip_name if out_dir.name == "Figures_Tables" else Path(zip_name)

    # Keep EXACT top-level folder name "Figures_Tables" inside the ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out_dir.rglob("*"):
            if p.is_file():
                arc = Path("Figures_Tables") / p.relative_to(out_dir)
                z.write(p, arcname=str(arc))
    return zip_path

def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default="./Figures_Tables")
    ap.add_argument("--with_meta", action="store_true", help="Runs meta-learner inside RQ5 (needs base weights).")
    args = ap.parse_args()

    scripts = [
        ["python", "src/rq1_backbone.py", "--data", args.data, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--out", args.out],
        ["python", "src/rq2_preprocessing.py", "--data", args.data, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--out", args.out],
        ["python", "src/rq3_ensemble.py", "--data", args.data, "--batch_size", str(args.batch_size), "--out", args.out],
        ["python", "src/rq4_rule_engine.py", "--data", args.data, "--out", args.out, "--model", "model_r50.pth"],
        ["python", "src/rq5_augmentation.py", "--data", args.data, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--out", args.out]
        + (["--run_meta"] if args.with_meta else []),
    ]

    for s in scripts:
        run(s)

    out_dir = Path(args.out)
    zp = zip_figures(out_dir, "Figures_Tables.zip")
    print("\nCreated:", zp)

if __name__ == "__main__":
    main()
