"""Audit all prediction parquets in results/predictions/.

Prints schema, descriptive stats, overall and per-fold correlation, and flags
suspicious files.  Non-zero exit code when any file is flagged.

Run from the project root:
    python scripts/audit_predictions.py

Exit codes:
    0  — no files flagged
    1  — at least one file flagged (|corr| > threshold)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PREDS_DIR = ROOT / "results" / "predictions"
CORR_FLAG_THRESHOLD = 0.20  # |corr| > this triggers a flag


def _describe_pred_file(path: Path) -> float:
    """Print audit for one parquet and return overall corr(pred, target)."""
    df = pd.read_parquet(path)

    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")
    print(f"  shape      : {df.shape}")
    print(f"  index type : {type(df.index).__name__}")
    if isinstance(df.index, pd.MultiIndex):
        levels = {n: df.index.get_level_values(n).nunique() for n in df.index.names}
        print(f"  index levels: {levels}")
    if "fold_id" in df.columns:
        print(f"  folds      : {sorted(df['fold_id'].unique().tolist())}")

    print()
    print("  target.describe():")
    print(df["target"].describe().to_string(float_format=lambda x: f"{x:.6f}"))

    print()
    print("  prediction.describe():")
    print(df["prediction"].describe().to_string(float_format=lambda x: f"{x:.6f}"))

    overall_corr = df["prediction"].corr(df["target"])
    flag_str = "  *** FLAGGED" if abs(overall_corr) > CORR_FLAG_THRESHOLD else ""
    print(f"\n  corr(pred, target) [overall] : {overall_corr:.4f}{flag_str}")

    # Per-fold correlation
    if "fold_id" in df.columns:
        print()
        print("  Per-fold corr(pred, target):")
        fold_corrs = []
        for fid, fdf in df.groupby("fold_id"):
            fc = fdf["prediction"].corr(fdf["target"])
            fold_corrs.append(fc)
            flag_f = " ***" if abs(fc) > CORR_FLAG_THRESHOLD else ""
            print(f"    fold {int(fid):3d}  corr={fc:+.4f}{flag_f}")
        fold_arr = np.array([c for c in fold_corrs if not np.isnan(c)])
        if len(fold_arr):
            print(f"  Per-fold corr stats: mean={fold_arr.mean():+.4f}  "
                  f"max_abs={np.abs(fold_arr).max():.4f}  "
                  f"n_flagged={int((np.abs(fold_arr) > CORR_FLAG_THRESHOLD).sum())}")

    # Naive hit rate (demeaned so threshold is above-median, matching reporting.py)
    preds = df["prediction"]
    if "fold_id" in df.columns:
        fold_med = df.groupby("fold_id")["prediction"].transform("median")
        preds = preds - fold_med
    hit = ((preds > 0) == (df["target"] > 0)).mean()
    print(f"\n  hit rate (demeaned) : {hit:.4f}")

    print()
    print("  head(3):")
    print(df.head(3).to_string())

    return overall_corr


def main() -> int:
    paths = sorted(PREDS_DIR.glob("*.parquet"))
    if not paths:
        print(f"No parquet files found in {PREDS_DIR}")
        return 1

    # Skip copies created by determinism tests
    paths = [p for p in paths if "run1" not in p.stem and "run2" not in p.stem]

    print(f"Auditing {len(paths)} prediction file(s) in {PREDS_DIR}\n")
    flagged = []
    for path in paths:
        corr = _describe_pred_file(path)
        if abs(corr) > CORR_FLAG_THRESHOLD:
            flagged.append((path.name, corr))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Files audited     : {len(paths)}")
    print(f"  Flag threshold    : |corr| > {CORR_FLAG_THRESHOLD}")
    if flagged:
        print(f"  FLAGGED ({len(flagged)})  — possible look-ahead bias:")
        for name, c in flagged:
            print(f"    {name}  overall_corr={c:.4f}")
        return 1
    else:
        print("  No files flagged.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

