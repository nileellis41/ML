"""Audit all prediction parquets in results/predictions/.

Prints schema, descriptive stats, correlation, and flags suspicious files.
Run from the project root:

    python scripts/audit_predictions.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PREDS_DIR = ROOT / "results" / "predictions"
CORR_FLAG_THRESHOLD = 0.20  # |corr| > this triggers a warning


def _describe_pred_file(path: Path) -> None:
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

    corr = df["prediction"].corr(df["target"])
    flag = "  *** FLAGGED: |corr| > threshold — possible look-ahead" if abs(corr) > CORR_FLAG_THRESHOLD else ""
    print(f"\n  corr(pred, target) : {corr:.4f}{flag}")

    hit = ((df["prediction"] > 0) == (df["target"] > 0)).mean()
    print(f"  naive hit rate     : {hit:.4f}  (sign(pred) == sign(target), all rows, incl. overlapping)")

    print()
    print("  head(3):")
    print(df.head(3).to_string())


def main() -> None:
    paths = sorted(PREDS_DIR.glob("*.parquet"))
    if not paths:
        print(f"No parquet files found in {PREDS_DIR}")
        sys.exit(1)

    print(f"Auditing {len(paths)} prediction file(s) in {PREDS_DIR}\n")
    flagged = []
    for path in paths:
        _describe_pred_file(path)
        df = pd.read_parquet(path)
        corr = df["prediction"].corr(df["target"])
        if abs(corr) > CORR_FLAG_THRESHOLD:
            flagged.append((path.name, corr))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Files audited : {len(paths)}")
    if flagged:
        print(f"  FLAGGED ({len(flagged)}):")
        for name, c in flagged:
            print(f"    {name}  corr={c:.4f}")
    else:
        print(f"  No files flagged (|corr| threshold = {CORR_FLAG_THRESHOLD})")


if __name__ == "__main__":
    main()
