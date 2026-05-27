"""Evaluation reporting: master comparison table + equity curve plots.

Produces:
  - results/tables/master_comparison.csv
  - results/figures/equity_curves_base.png
  - results/figures/equity_curves_regime.png
  - results/figures/drawdowns.png
  - results/figures/regime_shaded_spy.png

Entry point:
    python -m src.evaluation.reporting --run-all

Design notes
------------
Signal alignment: the prediction for date *t* is formed using features as of *t*
and predicts the 21-day forward return (t → t+21). To avoid overlapping windows,
we take ONE signal per walk-forward fold (the first date of each fold's test
period). The selected tickers are held for the entire fold duration (~21 trading
days). Portfolio return for each date in the fold is computed from actual daily
close prices in the Alpaca cache — giving genuine daily returns compatible with
metrics.py's 252-day annualisation formulas.

Hit-rate is computed over the same first-date-per-fold rows: sign(prediction)
vs sign(21-day target).
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.baselines.portfolios import (
    equal_weight,
    momentum_12_1,
    sixty_forty,
    spy_buy_hold,
)
from src.evaluation.metrics import (
    annualised_return,
    annualised_volatility,
    compute_all_metrics,
    sharpe_ratio,
    sharpe_se,
    stationary_bootstrap_sharpe,
)
from src.evaluation.deflated_sharpe import compute_dsr_for_returns
from src.utils.io import load_parquet
from src.utils.seeds import set_all_seeds

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path("results")
_TABLES_DIR = _RESULTS_DIR / "tables"
_FIGURES_DIR = _RESULTS_DIR / "figures"
_PREDS_DIR = _RESULTS_DIR / "predictions"
_ALPACA_CACHE = Path("data/raw/alpaca")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_predictions() -> dict[str, pd.DataFrame]:
    """Load all prediction parquets from results/predictions/."""
    preds = {}
    for path in sorted(_PREDS_DIR.glob("*.parquet")):
        key = path.stem
        df = load_parquet(path)
        if df is not None:
            preds[key] = df
    logger.info("Loaded %d prediction files", len(preds))
    return preds


def _load_daily_close(tickers: list[str]) -> pd.DataFrame:
    """Load daily close prices for *tickers* from the Alpaca cache.

    Returns a DataFrame (date × ticker) with tz-naive index.
    """
    frames = {}
    for ticker in tickers:
        path = _ALPACA_CACHE / f"{ticker}.parquet"
        df = load_parquet(path)
        if df is None or "close" not in df.columns:
            continue
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
            df = df.copy()
            df.index = idx
        frames[ticker] = df["close"]
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _first_date_signals(pred_df: pd.DataFrame) -> dict[int, list[str]]:
    """Return {fold_id: [long_tickers]} using the first date per fold that
    has at least one non-NaN prediction (handles LSTM burn-in NaNs).

    Signal rule: go long tickers whose prediction exceeds the cross-sectional
    median for that date.  This is model-agnostic — correct for both linear
    return predictions (which can be negative) and logistic probability outputs
    (which are always positive, making prediction>0 uninformative).
    """
    signals = {}
    for fold_id, fold_df in pred_df.groupby("fold_id"):
        dates = sorted(fold_df.index.get_level_values("date").unique())
        pos: list[str] = []
        for d in dates:
            day = fold_df[fold_df.index.get_level_values("date") == d]
            valid = day.dropna(subset=["prediction"])
            if not valid.empty:
                med = valid["prediction"].median()
                pos = (
                    valid[valid["prediction"] > med]
                    .index.get_level_values("ticker")
                    .tolist()
                )
                break
        signals[fold_id] = pos
    return signals


def _date_to_fold_map(pred_df: pd.DataFrame) -> dict:
    """Map each test date to its fold_id."""
    d2f = {}
    for fold_id, fold_df in pred_df.groupby("fold_id"):
        for d in fold_df.index.get_level_values("date").unique():
            d2f[d] = fold_id
    return d2f


# ---------------------------------------------------------------------------
# Portfolio return computation
# ---------------------------------------------------------------------------

def _returns_from_predictions(pred_df: pd.DataFrame) -> pd.Series:
    """Compute genuine daily portfolio returns from the prediction DataFrame.

    Algorithm
    ---------
    1. For each walk-forward fold, the signal is taken from the FIRST date only
       (avoids overlapping 21-day target windows).
    2. The selected tickers are held throughout the fold, earning actual daily
       log returns from the Alpaca close-price cache.
    3. Returns a daily Series compatible with metrics.py's annualisation (×252).

    This correctly separates prediction-frequency (monthly) from
    return-frequency (daily) and prevents inflated annualised metrics.
    """
    if "prediction" not in pred_df.columns:
        return pd.Series(dtype=float)
    if not isinstance(pred_df.index, pd.MultiIndex):
        return pd.Series(dtype=float)
    if "fold_id" not in pred_df.columns:
        return pd.Series(dtype=float)

    tickers = pred_df.index.get_level_values("ticker").unique().tolist()
    close = _load_daily_close(tickers)
    if close.empty:
        return pd.Series(dtype=float)

    # daily_ret[t] = log(close[t] / close[t-1]) — earned on day t
    daily_ret = np.log(close / close.shift(1))

    fold_signals = _first_date_signals(pred_df)
    date_to_fold = _date_to_fold_map(pred_df)

    test_dates = sorted(pred_df.index.get_level_values("date").unique())
    port: dict = {}
    for d in test_dates:
        fid = date_to_fold[d]
        pos = fold_signals.get(fid, [])
        avail = [
            t for t in pos
            if t in daily_ret.columns
            and d in daily_ret.index
            and not np.isnan(daily_ret.at[d, t])
        ]
        if not avail:
            port[d] = 0.0
        else:
            w = 1.0 / len(avail)
            port[d] = float(sum(w * daily_ret.at[d, t] for t in avail))

    result = pd.Series(port).sort_index()
    result.name = "portfolio_return"
    return result


def _hit_rate_pairs(pred_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract (predictions, actuals) for hit-rate using first date per fold.

    Uses 21-day targets: sign(prediction) vs sign(actual 21-day return).
    """
    if "prediction" not in pred_df.columns or "target" not in pred_df.columns:
        empty = pd.Series(dtype=float)
        return empty, empty
    if "fold_id" not in pred_df.columns or not isinstance(pred_df.index, pd.MultiIndex):
        empty = pd.Series(dtype=float)
        return empty, empty

    rows = []
    for fold_id, fold_df in pred_df.groupby("fold_id"):
        dates = sorted(fold_df.index.get_level_values("date").unique())
        for d in dates:
            day = fold_df[fold_df.index.get_level_values("date") == d]
            if day["prediction"].notna().any():
                rows.append(day)
                break

    if not rows:
        empty = pd.Series(dtype=float)
        return empty, empty

    signal_df = pd.concat(rows)
    valid = signal_df["prediction"].notna() & signal_df["target"].notna()

    # Demean predictions cross-sectionally within each fold so that
    # hit_rate's ">0" threshold means "above-median expected" rather
    # than "any positive output" (which is always True for logistic probs).
    preds = signal_df.loc[valid, "prediction"]
    fold_medians = preds.groupby(signal_df.loc[valid, "fold_id"]).transform("median")
    preds_demeaned = preds - fold_medians

    return preds_demeaned, signal_df.loc[valid, "target"]


def _spy_benchmark(test_dates: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Load SPY daily log returns for the test period."""
    close = _load_daily_close(["SPY"])
    if close.empty or "SPY" not in close.columns:
        return None
    ret = np.log(close["SPY"] / close["SPY"].shift(1))
    aligned = ret.reindex(test_dates)
    if aligned.isna().all():
        return None
    return aligned


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    predictions: dict[str, pd.DataFrame],
    regime_labels: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Compute all metrics for every model-variant and return as a DataFrame."""
    rows = []
    for model_variant, pred_df in predictions.items():
        parts = model_variant.rsplit("_", 1)
        model = parts[0] if len(parts) == 2 and parts[1] in {"base", "regime"} else model_variant
        variant = parts[1] if len(parts) == 2 and parts[1] in {"base", "regime"} else "base"

        port_returns = _returns_from_predictions(pred_df)
        if len(port_returns) == 0:
            logger.warning("No returns for %s -- skipping", model_variant)
            continue

        preds_series, actuals_series = _hit_rate_pairs(pred_df)
        test_dates = port_returns.index
        spy_ret = _spy_benchmark(test_dates)

        metrics = compute_all_metrics(
            portfolio_returns=port_returns,
            benchmark_returns=spy_ret,
            predictions=preds_series if not preds_series.empty else None,
            actuals=actuals_series if not actuals_series.empty else None,
            regime_labels=regime_labels,
            rf=rf,
        )
        row = {"model": model, "variant": variant, **metrics}
        rows.append(row)
        logger.info("Computed metrics for %s/%s", model, variant)

    if not rows:
        logger.warning("No valid model results to report.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(["model", "variant"])
    return df


# ---------------------------------------------------------------------------
# Per-fold metrics (Task 3)
# ---------------------------------------------------------------------------

def _per_fold_metrics(
    predictions: dict[str, pd.DataFrame],
    regime_labels: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Compute Sharpe ratio for every (model-variant, fold) combination.

    Returns a DataFrame with columns [model_variant, fold_id, sharpe, n_days, sharpe_se].
    """
    rows = []
    for model_variant, pred_df in predictions.items():
        if "fold_id" not in pred_df.columns:
            continue

        fold_signals = _first_date_signals(pred_df)
        date_to_fold = _date_to_fold_map(pred_df)

        tickers = pred_df.index.get_level_values("ticker").unique().tolist()
        close = _load_daily_close(tickers)
        if close.empty:
            continue
        daily_ret = np.log(close / close.shift(1))

        for fold_id in sorted(pred_df["fold_id"].unique()):
            fold_dates = sorted(
                pred_df[pred_df["fold_id"] == fold_id]
                .index.get_level_values("date")
                .unique()
            )
            pos = fold_signals.get(int(fold_id), [])

            port: dict = {}
            for d in fold_dates:
                avail = [
                    t for t in pos
                    if t in daily_ret.columns
                    and d in daily_ret.index
                    and not np.isnan(daily_ret.at[d, t])
                ]
                if avail:
                    port[d] = float(sum((1.0 / len(avail)) * daily_ret.at[d, t] for t in avail))
                else:
                    port[d] = 0.0

            fold_ret = pd.Series(port).sort_index()
            if len(fold_ret) < 5:
                continue

            sr = sharpe_ratio(fold_ret, rf=rf)
            se = sharpe_se(len(fold_ret), sr) if not np.isnan(sr) else np.nan
            rows.append({
                "model_variant": model_variant,
                "fold_id": int(fold_id),
                "sharpe": sr,
                "ann_return": annualised_return(fold_ret),
                "n_days": len(fold_ret),
                "sharpe_se": se,
            })

    return pd.DataFrame(rows)


def plot_per_fold_sharpe(per_fold_df: pd.DataFrame) -> None:
    """Boxplot of per-fold Sharpe distributions across model variants."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping per-fold Sharpe plot")
        return

    if per_fold_df.empty:
        return

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    variants = sorted(per_fold_df["model_variant"].unique())
    data = [per_fold_df[per_fold_df["model_variant"] == v]["sharpe"].dropna().values for v in variants]

    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 1.2), 6))
    bp = ax.boxplot(data, labels=variants, patch_artist=True, notch=False)
    colors = plt.cm.tab20.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Per-Fold Annualised Sharpe Distribution by Model")
    ax.set_xlabel("Model / Variant")
    ax.set_ylabel("Annualised Sharpe (30-day fold)")
    ax.set_xticklabels(variants, rotation=35, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / "sharpe_per_fold.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Per-fold Sharpe boxplot saved: %s", out)


# ---------------------------------------------------------------------------
# Baselines (Task 2)
# ---------------------------------------------------------------------------

def build_baselines_rows(
    test_dates: pd.DatetimeIndex,
    tickers: list[str],
    spy_ret: Optional[pd.Series],
    regime_labels: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> list[dict]:
    """Compute metrics for the 4 passive/rule-based baselines."""
    baselines = {
        "spy_bh": spy_buy_hold(test_dates),
        "equal_weight": equal_weight(test_dates, tickers),
        "momentum_12_1": momentum_12_1(test_dates, tickers),
        "sixty_forty": sixty_forty(test_dates),
    }
    rows = []
    for name, ret in baselines.items():
        if ret is None or ret.isna().all():
            logger.warning("Baseline %s returned all-NaN; skipping", name)
            continue
        ret = ret.fillna(0.0)
        m = compute_all_metrics(
            portfolio_returns=ret,
            benchmark_returns=spy_ret,
            regime_labels=regime_labels,
            rf=rf,
        )
        rows.append({"model": name, "variant": "baseline", **m})
        logger.info("Computed metrics for baseline/%s", name)
    return rows


# ---------------------------------------------------------------------------
# Table persistence
# ---------------------------------------------------------------------------

def _format_regime_sharpe_col(table: pd.DataFrame, regime: str) -> pd.Series:
    """Render 'X.XX [lo, hi]' for a regime Sharpe column when n_periods < 100."""
    sharpe_col = f"regime_{regime}_sharpe"
    se_col = f"regime_{regime}_sharpe_se"
    lo_col = f"regime_{regime}_sharpe_ci_lo"
    hi_col = f"regime_{regime}_sharpe_ci_hi"
    n_col = f"regime_{regime}_n_periods"

    if sharpe_col not in table.columns:
        return None

    def _fmt(row):
        sr = row.get(sharpe_col, float("nan"))
        n = row.get(n_col, 999)
        lo = row.get(lo_col, float("nan"))
        hi = row.get(hi_col, float("nan"))
        if pd.isna(sr):
            return "nan"
        if not pd.isna(lo) and not pd.isna(hi) and n < 100:
            return f"{sr:.2f} [{lo:.2f}, {hi:.2f}]"
        return f"{sr:.2f}"

    return table.apply(_fmt, axis=1)


def _metadata_header() -> str:
    """Return a one-line metadata comment for table headers."""
    import datetime, sys, subprocess
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    pyver = f"Python {sys.version.split()[0]}"
    try:
        commit = subprocess.check_output(
            ["git", "log", "--oneline", "-1"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        commit = "git-not-available"
    return f"# Generated: {ts} | {pyver} | commit: {commit}\n"


def save_tables(table: pd.DataFrame) -> None:
    _TABLES_DIR.mkdir(parents=True, exist_ok=True)
    meta = _metadata_header()
    csv_path = _TABLES_DIR / "master_comparison.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(meta)
    table.to_csv(csv_path, mode="a")
    logger.info("Master comparison table saved to %s", csv_path)

    # Build a display copy that merges Sharpe+CI into one column
    display = table.copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].round(4)

    # Format main Sharpe as "X.XX [lo, hi]" when bootstrap CIs are present
    if "sharpe_ci_lo" in table.columns and "sharpe_ci_hi" in table.columns:
        def _fmt_main_sharpe(row):
            sr = row.get("sharpe", float("nan"))
            lo = row.get("sharpe_ci_lo", float("nan"))
            hi = row.get("sharpe_ci_hi", float("nan"))
            if pd.isna(sr):
                return "nan"
            if not pd.isna(lo) and not pd.isna(hi):
                return f"{sr:.2f} [{lo:.2f}, {hi:.2f}]"
            return f"{sr:.2f}"
        display["sharpe"] = table.apply(_fmt_main_sharpe, axis=1)
        display = display.drop(columns=["sharpe_ci_lo", "sharpe_ci_hi"], errors="ignore")

    # Inline CI into regime Sharpe columns; drop raw se/ci_lo/ci_hi columns
    regime_labels_found = set()
    for col in table.columns:
        if col.startswith("regime_") and col.endswith("_sharpe"):
            label = "_".join(col.split("_")[1:-1])
            regime_labels_found.add(label)

    ci_cols_to_drop = []
    for regime in regime_labels_found:
        formatted = _format_regime_sharpe_col(table, regime)
        if formatted is not None:
            display[f"regime_{regime}_sharpe"] = formatted
        for suffix in ("_sharpe_se", "_sharpe_ci_lo", "_sharpe_ci_hi"):
            col = f"regime_{regime}{suffix}"
            if col in display.columns:
                ci_cols_to_drop.append(col)
    display = display.drop(columns=ci_cols_to_drop, errors="ignore")

    md_path = _TABLES_DIR / "master_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master Model Comparison\n\n")
        f.write(f"> {meta.lstrip('# ').strip()}\n\n")
        f.write("> Sharpe shown as: value [95% bootstrap CI lo, hi]\n\n")
        f.write("> Regime Sharpe shown as: value [95% CI lo, hi] when n_periods < 100\n\n")
        f.write(display.to_markdown())
        f.write("\n")
    logger.info("Markdown table saved to %s", md_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_curves_final(
    ml_returns: dict[str, pd.Series],
    baseline_returns: dict[str, pd.Series],
    regime_labels: Optional[pd.Series] = None,
) -> None:
    """All-12-strategy equity curves figure for the final deliverable.

    ML strategies: solid lines.  Baselines: dashed lines.
    Colorblind-safe Okabe-Ito palette.  Regime shading at alpha=0.1.
    Saves .png and .svg at dpi=150.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available; skipping equity_curves_final")
        return

    # Okabe-Ito palette (8 colors, colorblind-safe)
    _OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
           "#0072B2", "#D55E00", "#CC79A7", "#000000",
           "#999999", "#44AA99", "#882255", "#DDCC77"]

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    color_idx = 0
    for key in sorted(ml_returns):
        ret = ml_returns[key]
        cum = ret.fillna(0.0).cumsum()
        ax.plot(cum.index, cum.values, label=key,
                color=_OI[color_idx % len(_OI)], linewidth=1.4, alpha=0.85)
        color_idx += 1

    for key in sorted(baseline_returns):
        ret = baseline_returns[key]
        cum = ret.fillna(0.0).cumsum()
        ax.plot(cum.index, cum.values, label=key,
                color=_OI[color_idx % len(_OI)], linewidth=1.8,
                linestyle="--", alpha=0.9)
        color_idx += 1

    # Light regime shading
    if regime_labels is not None:
        regime_palette = {
            "risk_on": "#009E73", "neutral": "#56B4E9",
            "risk_off": "#E69F00", "crisis": "#D55E00",
        }
        # Build contiguous spans
        all_dates = sorted(
            set().union(*(r.index.tolist() for r in ml_returns.values()))
        )
        if all_dates:
            rl = regime_labels.reindex(all_dates, method="ffill")
            current, start = None, None
            for d, lbl in rl.items():
                if lbl != current:
                    if current is not None:
                        ax.axvspan(start, d, alpha=0.08,
                                   color=regime_palette.get(str(current), "#aaaaaa"),
                                   linewidth=0)
                    current, start = lbl, d
            if current is not None:
                ax.axvspan(start, rl.index[-1], alpha=0.08,
                           color=regime_palette.get(str(current), "#aaaaaa"),
                           linewidth=0)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "svg"):
        out = _FIGURES_DIR / f"equity_curves_final.{ext}"
        fig.savefig(out, dpi=150)
        logger.info("Final equity curve plot saved: %s", out)
    plt.close(fig)


def plot_sharpe_per_fold_final(per_fold_df: pd.DataFrame) -> None:
    """Publication-quality per-fold Sharpe boxplot with baselines.

    Shows 12 boxes (8 ML + 4 baselines), a zero line, and a reference line
    at the median per-fold Sharpe of momentum_12_1.
    Saves .png and .svg at dpi=150.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping sharpe_per_fold_final")
        return

    if per_fold_df.empty:
        return

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Order: ML models first (alphabetical), then baselines
    all_variants = sorted(per_fold_df["model_variant"].unique())
    ml_variants = [v for v in all_variants if not any(
        v.startswith(b) for b in ("spy_bh", "equal_weight", "momentum_12_1", "sixty_forty")
    )]
    bl_variants = [v for v in all_variants if v not in ml_variants]
    ordered = ml_variants + bl_variants

    data = [per_fold_df[per_fold_df["model_variant"] == v]["sharpe"].dropna().values
            for v in ordered]

    # Median per-fold Sharpe for momentum_12_1
    mom_med = np.nan
    mom_key = next((v for v in ordered if "momentum" in v), None)
    if mom_key:
        mom_data = per_fold_df[per_fold_df["model_variant"] == mom_key]["sharpe"].dropna()
        if not mom_data.empty:
            mom_med = float(mom_data.median())

    fig, ax = plt.subplots(figsize=(max(12, len(ordered) * 1.3), 6))

    # Colour: ML = light blue, baselines = light orange
    n_ml = len(ml_variants)
    colors = ["#56B4E9"] * n_ml + ["#E69F00"] * len(bl_variants)

    bp = ax.boxplot(data, tick_labels=ordered, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
               label="Sharpe = 0")
    if not np.isnan(mom_med):
        ax.axhline(mom_med, color="#D55E00", linewidth=1.2, linestyle=":",
                   alpha=0.8, label=f"momentum_12_1 median = {mom_med:.2f}")

    ax.set_xlabel("Model / Variant")
    ax.set_ylabel("Annualised Sharpe (per fold)")
    ax.set_xticklabels(ordered, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "svg"):
        out = _FIGURES_DIR / f"sharpe_per_fold_final.{ext}"
        fig.savefig(out, dpi=150)
        logger.info("Final per-fold Sharpe boxplot saved: %s", out)
    plt.close(fig)


def plot_equity_curves(
    predictions: dict[str, pd.DataFrame],
    suffix: str = "base",
) -> None:
    """Overlay equity curves for all models with a given variant suffix, plus SPY B&H."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping equity curve plot")
        return

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0
    all_dates = pd.DatetimeIndex([])

    for model_variant, pred_df in predictions.items():
        if not model_variant.endswith(f"_{suffix}"):
            continue
        port_returns = _returns_from_predictions(pred_df)
        if len(port_returns) == 0:
            continue
        all_dates = all_dates.union(port_returns.index)
        cum = port_returns.cumsum()
        ax.plot(cum.index, cum.values, label=model_variant, alpha=0.8)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    # SPY buy-and-hold reference
    if not all_dates.empty:
        spy_ret = _spy_benchmark(all_dates)
        if spy_ret is not None:
            cum_spy = spy_ret.fillna(0.0).cumsum()
            ax.plot(
                cum_spy.index, cum_spy.values,
                label="SPY B&H", color="black", linewidth=2,
                linestyle="--", alpha=0.7,
            )

    ax.set_title(f"Equity Curves — {suffix.title()} Models (cumulative log return)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / f"equity_curves_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Equity curve plot saved: %s", out)


def plot_regime_shaded_spy(
    spy_returns: pd.Series,
    regime_labels: pd.Series,
) -> None:
    """Plot SPY cumulative return with regime-coloured background."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available; skipping regime shaded plot")
        return

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    palette = {
        "risk_on": "#2ca02c",
        "neutral": "#1f77b4",
        "risk_off": "#ff7f0e",
        "crisis": "#d62728",
    }

    common = spy_returns.index.intersection(regime_labels.index)
    ret = spy_returns.reindex(common)
    labels = regime_labels.reindex(common)
    cum = ret.cumsum()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(cum.index, cum.values, color="black", linewidth=1.5, label="SPY")

    current_label, start_date = None, None
    for date, label in zip(labels.index, labels.values):
        if label != current_label:
            if current_label is not None and start_date is not None:
                ax.axvspan(start_date, date, alpha=0.15, color=palette.get(str(current_label), "#aaaaaa"))
            current_label = label
            start_date = date
    if current_label is not None and start_date is not None:
        ax.axvspan(start_date, labels.index[-1], alpha=0.15, color=palette.get(str(current_label), "#aaaaaa"))

    patches = [mpatches.Patch(color=c, alpha=0.3, label=l) for l, c in palette.items()]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="black", label="SPY")], fontsize=9)
    ax.set_title("SPY Cumulative Return with Regime Background")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out = _FIGURES_DIR / "regime_shaded_spy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Regime-shaded SPY plot saved: %s", out)


def plot_drawdowns(predictions: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    for model_variant, pred_df in predictions.items():
        port_returns = _returns_from_predictions(pred_df)
        if len(port_returns) == 0:
            continue
        cum = port_returns.cumsum()
        drawdown = cum - cum.cummax()
        ax.plot(drawdown.index, drawdown.values, label=model_variant, alpha=0.7)

    ax.set_title("Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (cumulative log return)")
    ax.legend(loc="lower left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / "drawdowns.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Drawdown plot saved: %s", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(rf: float = 0.0) -> pd.DataFrame:
    """Load all cached predictions, compute metrics, save tables and plots."""
    import time
    _t0 = time.time()
    set_all_seeds()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    predictions = _load_predictions()
    if not predictions:
        logger.error("No prediction files found in %s. Run experiments first.", _PREDS_DIR)
        return pd.DataFrame()

    regime_labels: Optional[pd.Series] = None
    hmm_path = Path("data/regimes/hmm_probs.parquet")
    if hmm_path.exists():
        hmm_df = load_parquet(hmm_path)
        if hmm_df is not None and "predicted_label" in hmm_df.columns:
            regime_labels = hmm_df["predicted_label"]

    table = build_comparison_table(predictions, regime_labels=regime_labels, rf=rf)

    # --- Baselines ---
    first_pred = next(iter(predictions.values()))
    test_dates = first_pred.index.get_level_values("date").unique().sort_values()
    all_tickers = first_pred.index.get_level_values("ticker").unique().tolist()
    spy_ret_full = _spy_benchmark(test_dates)

    baseline_ret_series = {
        "spy_bh":        spy_buy_hold(test_dates).fillna(0.0),
        "equal_weight":  equal_weight(test_dates, all_tickers).fillna(0.0),
        "momentum_12_1": momentum_12_1(test_dates, all_tickers).fillna(0.0),
        "sixty_forty":   sixty_forty(test_dates).fillna(0.0),
    }
    baseline_rows = build_baselines_rows(
        test_dates, all_tickers, spy_ret_full, regime_labels=regime_labels, rf=rf
    )
    if baseline_rows:
        baseline_df = pd.DataFrame(baseline_rows).set_index(["model", "variant"])
        table = pd.concat([table, baseline_df])

    # --- Bootstrap Sharpe CIs (Task 2) + Deflated Sharpe (Task 5) ---
    logger.info("Computing bootstrap Sharpe CIs and DSR for %d strategies ...", len(table))
    ml_ret_series: dict[str, pd.Series] = {}
    for key, pred_df in predictions.items():
        ret = _returns_from_predictions(pred_df)
        if len(ret) > 0:
            ml_ret_series[key] = ret

    all_ret_series: dict[str, pd.Series] = {**ml_ret_series, **baseline_ret_series}

    def _index_key(model: str, variant: str) -> str:
        if variant == "baseline":
            return model
        return f"{model}_{variant}"

    for (model, variant) in table.index:
        ret_key = _index_key(model, variant)
        ret = all_ret_series.get(ret_key)
        if ret is None:
            continue
        _, ci_lo, ci_hi = stationary_bootstrap_sharpe(ret, n_boot=10_000)
        dsr_pval = compute_dsr_for_returns(ret, n_trials=8)
        table.at[(model, variant), "sharpe_ci_lo"] = ci_lo
        table.at[(model, variant), "sharpe_ci_hi"] = ci_hi
        table.at[(model, variant), "deflated_sharpe_pvalue"] = dsr_pval

    if not table.empty:
        save_tables(table)

    # --- Per-fold metrics with baselines (Tasks 3 & 4) ---
    per_fold_df = _per_fold_metrics(predictions, regime_labels=regime_labels, rf=rf)

    # Add baseline per-fold Sharpe using ML fold date ranges
    if not per_fold_df.empty:
        fold_dates_map: dict[int, list] = {}
        for key, pred_df in predictions.items():
            if "fold_id" not in pred_df.columns:
                continue
            for fid, fdf in pred_df.groupby("fold_id"):
                fold_dates_map[int(fid)] = sorted(
                    fdf.index.get_level_values("date").unique()
                )
            break  # only need one model's fold structure

        bl_rows = []
        for bl_name, bl_ret in baseline_ret_series.items():
            for fid, dates in fold_dates_map.items():
                slice_ret = bl_ret.reindex(dates).dropna()
                if len(slice_ret) < 5:
                    continue
                sr = sharpe_ratio(slice_ret, rf=rf)
                se = sharpe_se(len(slice_ret), sr) if not np.isnan(sr) else np.nan
                bl_rows.append({
                    "model_variant": bl_name,
                    "fold_id": fid,
                    "sharpe": sr,
                    "ann_return": annualised_return(slice_ret),
                    "n_days": len(slice_ret),
                    "sharpe_se": se,
                })
        if bl_rows:
            per_fold_df = pd.concat([per_fold_df, pd.DataFrame(bl_rows)], ignore_index=True)

        _TABLES_DIR.mkdir(parents=True, exist_ok=True)
        per_fold_df.to_csv(_TABLES_DIR / "per_fold_summary.csv", index=False)
        logger.info("Per-fold summary saved: %d rows", len(per_fold_df))
        plot_per_fold_sharpe(per_fold_df)
        plot_sharpe_per_fold_final(per_fold_df)

    # --- Final equity curves figure (Task 3) ---
    plot_equity_curves_final(ml_ret_series, baseline_ret_series, regime_labels)

    # --- Legacy per-variant equity curves ---
    plot_equity_curves(predictions, suffix="base")
    plot_equity_curves(predictions, suffix="regime")
    plot_drawdowns(predictions)

    spy_path = _ALPACA_CACHE / "SPY.parquet"
    if spy_path.exists() and regime_labels is not None:
        spy_df = load_parquet(spy_path)
        if spy_df is not None and "close" in spy_df.columns:
            if spy_df.index.tz is not None:
                spy_df.index = spy_df.index.tz_convert(None)
            spy_ret = np.log(spy_df["close"] / spy_df["close"].shift(1)).dropna()
            plot_regime_shaded_spy(spy_ret, regime_labels)

    wall_time = time.time() - _t0
    logger.info("Reporting complete in %.1f s. Results in %s", wall_time, _RESULTS_DIR)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF Research Reporting")
    parser.add_argument("--run-all", action="store_true", help="Generate all reports from cached predictions")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate")
    args = parser.parse_args()
    if args.run_all:
        run_all(rf=args.rf)
