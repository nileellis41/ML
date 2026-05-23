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

from src.evaluation.metrics import compute_all_metrics
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
    """Return {fold_id: [positive_tickers]} using the first date per fold
    that has at least one non-NaN prediction (handles LSTM burn-in NaNs)."""
    signals = {}
    for fold_id, fold_df in pred_df.groupby("fold_id"):
        dates = sorted(fold_df.index.get_level_values("date").unique())
        pos: list[str] = []
        for d in dates:
            day = fold_df[fold_df.index.get_level_values("date") == d]
            valid = day.dropna(subset=["prediction"])
            if not valid.empty:
                pos = (
                    valid[valid["prediction"] > 0]
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
    return signal_df.loc[valid, "prediction"], signal_df.loc[valid, "target"]


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
# Table persistence
# ---------------------------------------------------------------------------

def save_tables(table: pd.DataFrame) -> None:
    _TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _TABLES_DIR / "master_comparison.csv"
    table.to_csv(csv_path)
    logger.info("Master comparison table saved to %s", csv_path)

    md_path = _TABLES_DIR / "master_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Master Model Comparison\n\n")
        display = table.copy()
        for col in display.select_dtypes(include="number").columns:
            display[col] = display[col].round(4)
        f.write(display.to_markdown())
        f.write("\n")
    logger.info("Markdown table saved to %s", md_path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

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
    if not table.empty:
        save_tables(table)

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

    logger.info("Reporting complete. Results in %s", _RESULTS_DIR)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF Research Reporting")
    parser.add_argument("--run-all", action="store_true", help="Generate all reports from cached predictions")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate")
    args = parser.parse_args()
    if args.run_all:
        run_all(rf=args.rf)
