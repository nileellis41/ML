"""Hit-rate diagnostic: break down by calendar year, ticker, and HMM regime.

Flags cells where hit rate > 70 % (suspiciously good) or < 45 % (worse than
random), using a one-tailed binomial p-value for significance.

Saves results/tables/hit_rate_audit.csv.

Run from project root:
    python scripts/audit_hit_rate.py

Exit codes:
    0  — no flags
    1  — at least one flagged cell
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

PREDS_DIR = ROOT / "results" / "predictions"
REGIMES_PATH = ROOT / "data" / "regimes" / "hmm_probs.parquet"
OUT_PATH = ROOT / "results" / "tables" / "hit_rate_audit.csv"

FLAG_HI = 0.70   # flag if hit rate above this
FLAG_LO = 0.45   # flag if hit rate below this
MIN_OBS = 10     # minimum observations to include a cell


def _hit_rate_with_pval(pred: pd.Series, actual: pd.Series) -> tuple[float, float, int]:
    """Return (hit_rate, binomial_p_two_tailed, n)."""
    valid = pred.notna() & actual.notna()
    n = int(valid.sum())
    if n < MIN_OBS:
        return np.nan, np.nan, n
    hits = int(((pred[valid] > 0) == (actual[valid] > 0)).sum())
    hr = hits / n
    # Two-tailed binomial test vs p=0.5
    p = float(stats.binomtest(hits, n, p=0.5, alternative="two-sided").pvalue)
    return hr, p, n


def _load_regime_labels() -> pd.Series:
    if not REGIMES_PATH.exists():
        return pd.Series(dtype=str)
    df = pd.read_parquet(REGIMES_PATH)
    return df.get("predicted_label", pd.Series(dtype=str))


def _process_file(path: Path, regime_labels: pd.Series) -> pd.DataFrame:
    pred_df = pd.read_parquet(path)
    model_name = path.stem

    if "prediction" not in pred_df.columns or "target" not in pred_df.columns:
        return pd.DataFrame()
    if not isinstance(pred_df.index, pd.MultiIndex):
        return pd.DataFrame()

    # Demean predictions within fold (same as reporting.py) so >0 = above-median
    if "fold_id" in pred_df.columns:
        fold_medians = pred_df.groupby("fold_id")["prediction"].transform("median")
        pred_df = pred_df.copy()
        pred_df["prediction"] = pred_df["prediction"] - fold_medians

    dates = pred_df.index.get_level_values("date")
    tickers = pred_df.index.get_level_values("ticker")

    rows = []

    # ---- By calendar year ----
    years = dates.year.unique()
    for yr in sorted(years):
        mask = dates.year == yr
        hr, pval, n = _hit_rate_with_pval(
            pred_df.loc[mask, "prediction"], pred_df.loc[mask, "target"]
        )
        rows.append({
            "model": model_name, "dimension": "year", "group": str(yr),
            "hit_rate": hr, "pval": pval, "n": n,
        })

    # ---- By ticker ----
    for t in sorted(tickers.unique()):
        mask = tickers == t
        hr, pval, n = _hit_rate_with_pval(
            pred_df.loc[mask, "prediction"], pred_df.loc[mask, "target"]
        )
        rows.append({
            "model": model_name, "dimension": "ticker", "group": t,
            "hit_rate": hr, "pval": pval, "n": n,
        })

    # ---- By HMM regime ----
    if not regime_labels.empty:
        for idx in pred_df.index:
            d = idx[0]
            if d not in dates:
                pass
        # Map dates → regime label
        date_labels = regime_labels.reindex(dates)
        for label in sorted(date_labels.dropna().unique()):
            mask = date_labels == label
            hr, pval, n = _hit_rate_with_pval(
                pred_df.loc[mask.values, "prediction"],
                pred_df.loc[mask.values, "target"],
            )
            rows.append({
                "model": model_name, "dimension": "regime", "group": str(label),
                "hit_rate": hr, "pval": pval, "n": n,
            })

    df = pd.DataFrame(rows)
    df["flagged"] = (
        (df["hit_rate"] > FLAG_HI) | (df["hit_rate"] < FLAG_LO)
    ) & (df["n"] >= MIN_OBS)
    df["flag_reason"] = ""
    df.loc[df["hit_rate"] > FLAG_HI, "flag_reason"] = f"hit_rate > {FLAG_HI}"
    df.loc[df["hit_rate"] < FLAG_LO, "flag_reason"] = f"hit_rate < {FLAG_LO}"
    return df


def main() -> int:
    paths = sorted(PREDS_DIR.glob("*.parquet"))
    if not paths:
        print(f"No parquet files found in {PREDS_DIR}")
        return 1

    regime_labels = _load_regime_labels()
    all_rows = []

    for path in paths:
        df = _process_file(path, regime_labels)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("No valid prediction files processed.")
        return 1

    result = pd.concat(all_rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"Hit-rate audit saved: {OUT_PATH}  ({len(result)} cells)")

    flagged = result[result["flagged"]]
    total_flags = len(flagged)

    # ---- Print summary ----
    print()
    print("=" * 70)
    print(f"HIT-RATE AUDIT SUMMARY  (flags: HR > {FLAG_HI} or < {FLAG_LO}, n >= {MIN_OBS})")
    print("=" * 70)

    if total_flags == 0:
        print("No cells flagged.")
    else:
        print(f"FLAGGED CELLS ({total_flags}):")
        for _, row in flagged.sort_values(["model", "dimension", "group"]).iterrows():
            hr = f"{row['hit_rate']:.3f}" if not np.isnan(row["hit_rate"]) else "nan"
            pv = f"{row['pval']:.4f}" if not np.isnan(row["pval"]) else "nan"
            print(
                f"  {row['model']:<30} {row['dimension']:<8} {str(row['group']):<14}"
                f"  HR={hr}  p={pv}  n={row['n']:4d}  [{row['flag_reason']}]"
            )

    print()
    print("Overall hit rates by model (full-sample, demeaned predictions):")
    for model, gdf in result.groupby("model"):
        # Reconstruct overall from year rows for comparability
        yr_rows = gdf[gdf["dimension"] == "year"]
        total_n = yr_rows["n"].sum()
        avg_hr = (yr_rows["hit_rate"] * yr_rows["n"]).sum() / total_n if total_n > 0 else np.nan
        print(f"  {model:<30}  overall_HR={avg_hr:.3f}  n={total_n}")

    return 1 if total_flags > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
