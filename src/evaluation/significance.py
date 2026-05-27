"""Diebold-Mariano significance tests for portfolio return comparisons.

Compares each ML model's daily portfolio returns against each of four passive
baselines using the Diebold-Mariano (1995) test with the Harvey-Leybourne-
Newbold (1997) small-sample correction.  Applies Bonferroni and Benjamini-
Hochberg multiple-comparison adjustments across the full set of comparisons.

References
----------
Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
    prediction mean squared errors. International Journal of Forecasting, 13,
    281-291.
"""
import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def diebold_mariano_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    h: int = 1,
) -> tuple[float, float, float, float]:
    """DM test (power=1) with Harvey-Leybourne-Newbold small-sample correction.

    Null: E[r_a - r_b] = 0  (no systematic return difference).
    Alternative: two-tailed.

    Parameters
    ----------
    returns_a : daily log return series for strategy A (ML model)
    returns_b : daily log return series for strategy B (baseline)
    h : forecast horizon in periods (1 for daily returns)

    Returns
    -------
    (mean_return_diff_annual, sharpe_diff, mdm_stat, pvalue_twotailed)
        All NaN if fewer than 20 common observations.
    """
    common = returns_a.index.intersection(returns_b.index)
    if len(common) < 20:
        return np.nan, np.nan, np.nan, np.nan

    a = returns_a.reindex(common).fillna(0.0)
    b = returns_b.reindex(common).fillna(0.0)
    d = a - b  # signed return differential (power=1)
    T = len(d)

    d_bar = float(d.mean())
    var_d = float(d.var(ddof=1))

    if var_d <= 0:
        # Identically equal series: d is all zeros, DM stat = 0, p = 1
        if abs(d_bar) < 1e-15:
            return 0.0, 0.0, 0.0, 1.0
        return float(d_bar * _TRADING_DAYS), np.nan, np.nan, np.nan

    # Standard DM statistic (equivalent to a t-test on the differential)
    dm = d_bar / np.sqrt(var_d / T)

    # Harvey-Leybourne-Newbold (1997) small-sample correction:
    # MDM = sqrt( (T + 1 - 2h + h*(h-1)/T) / T ) * DM
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    mdm = correction * dm

    # Under H0, MDM ~ t(T-1) approximately
    pval = float(2.0 * stats.t.sf(abs(mdm), df=T - 1))

    # Sharpe difference on common dates
    std_a, std_b = float(a.std(ddof=1)), float(b.std(ddof=1))
    sr_a = a.mean() / std_a * np.sqrt(_TRADING_DAYS) if std_a > 0 else np.nan
    sr_b = b.mean() / std_b * np.sqrt(_TRADING_DAYS) if std_b > 0 else np.nan
    sharpe_diff = float(sr_a - sr_b) if not (np.isnan(sr_a) or np.isnan(sr_b)) else np.nan

    return float(d_bar * _TRADING_DAYS), sharpe_diff, float(mdm), pval


def run_significance_tests(
    ml_returns: dict[str, pd.Series],
    baseline_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """Run DM tests for all ML × baseline pairs with multiple-comparison corrections.

    Parameters
    ----------
    ml_returns : {model_variant_key: daily_return_series}  (8 ML configs)
    baseline_returns : {baseline_name: daily_return_series}  (4 baselines)

    Returns
    -------
    DataFrame with 32 rows and columns:
        ml_model, ml_variant, baseline, mean_return_diff_annual, sharpe_diff,
        dm_stat, pvalue_raw, pvalue_bonferroni, pvalue_bh,
        significant_at_05_after_bonferroni
    """
    rows = []
    for ml_key, ml_ret in sorted(ml_returns.items()):
        parts = ml_key.rsplit("_", 1)
        ml_model = parts[0] if len(parts) == 2 and parts[1] in {"base", "regime"} else ml_key
        ml_variant = parts[1] if len(parts) == 2 and parts[1] in {"base", "regime"} else "base"

        for bl_name, bl_ret in sorted(baseline_returns.items()):
            mean_diff, sharpe_diff, dm_stat, pval = diebold_mariano_test(ml_ret, bl_ret)
            rows.append({
                "ml_model": ml_model,
                "ml_variant": ml_variant,
                "baseline": bl_name,
                "mean_return_diff_annual": mean_diff,
                "sharpe_diff": sharpe_diff,
                "dm_stat": dm_stat,
                "pvalue_raw": pval,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    n_comparisons = len(df)
    raw_pvals = df["pvalue_raw"].fillna(1.0).values

    # Bonferroni: multiply by number of comparisons, cap at 1
    df["pvalue_bonferroni"] = np.minimum(raw_pvals * n_comparisons, 1.0)

    # Benjamini-Hochberg FDR
    _, pvals_bh, _, _ = multipletests(raw_pvals, method="fdr_bh")
    df["pvalue_bh"] = pvals_bh

    # Restore NaN for rows where the raw test was undefined
    nan_mask = df["pvalue_raw"].isna()
    df.loc[nan_mask, "pvalue_bonferroni"] = np.nan
    df.loc[nan_mask, "pvalue_bh"] = np.nan

    df["significant_at_05_after_bonferroni"] = df["pvalue_bonferroni"] < 0.05

    return df


def format_significance_summary(df: pd.DataFrame) -> str:
    """Render a human-readable significance summary table."""
    n_total = len(df)
    n_sig_bonf = int(df["significant_at_05_after_bonferroni"].sum())
    n_sig_bh = int((df["pvalue_bh"] < 0.05).sum())

    lines = [
        "# Significance Test Summary",
        "",
        f"ML vs baseline pairs tested : {n_total}  (8 ML configs x 4 baselines)",
        f"Bonferroni threshold        : p < {0.05 / n_total:.5f}  (alpha=0.05 / {n_total})",
        f"Significant after Bonferroni: {n_sig_bonf} / {n_total}",
        f"Significant after BH FDR    : {n_sig_bh} / {n_total}",
        "",
        "## Per-comparison results  (sorted by raw p-value)",
        "",
    ]

    header = (
        f"{'ML model':<18} {'variant':<8} {'baseline':<14}"
        f"  {'dReturn':>9}  {'dSharpe':>8}  {'DM':>7}  {'p_raw':>8}"
        f"  {'p_bonf':>8}  {'p_BH':>8}  {'sig*':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in df.sort_values("pvalue_raw").iterrows():
        sig = "YES" if row["significant_at_05_after_bonferroni"] else "no"
        p_raw = f"{row['pvalue_raw']:.4f}" if not pd.isna(row["pvalue_raw"]) else "  nan"
        p_bonf = f"{row['pvalue_bonferroni']:.4f}" if not pd.isna(row["pvalue_bonferroni"]) else "  nan"
        p_bh = f"{row['pvalue_bh']:.4f}" if not pd.isna(row["pvalue_bh"]) else "  nan"
        dm = f"{row['dm_stat']:+.3f}" if not pd.isna(row["dm_stat"]) else "  nan"
        dr = f"{row['mean_return_diff_annual']:+.4f}" if not pd.isna(row["mean_return_diff_annual"]) else "  nan"
        ds = f"{row['sharpe_diff']:+.3f}" if not pd.isna(row["sharpe_diff"]) else "  nan"
        lines.append(
            f"{row['ml_model']:<18} {row['ml_variant']:<8} {row['baseline']:<14}"
            f"  {dr:>9}  {ds:>8}  {dm:>7}  {p_raw:>8}"
            f"  {p_bonf:>8}  {p_bh:>8}  {sig:>5}"
        )

    lines += [
        "",
        "\\* Bonferroni-corrected at α=0.05",
        "",
        "## Bottom line",
        "",
    ]

    if n_sig_bonf == 0:
        lines.append(
            f"After Bonferroni correction across {n_total} comparisons, "
            f"**0 of {n_total} ML-vs-baseline pairs are significant at p<0.05**. "
            "We cannot reject the null hypothesis that the ML models produce "
            "the same expected returns as the passive baselines."
        )
    else:
        lines.append(
            f"After Bonferroni correction, **{n_sig_bonf} of {n_total} pairs are significant**. "
            "See flagged rows above."
        )

    return "\n".join(lines)
