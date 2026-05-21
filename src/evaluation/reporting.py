"""Evaluation reporting: master comparison table + equity curve plots.

Produces:
  - results/tables/master_comparison.csv
  - results/figures/equity_curves_base.png
  - results/figures/equity_curves_regime.png
  - results/figures/drawdowns.png
  - results/figures/regime_shaded_spy.png
  - results/figures/prediction_errors.png

Entry point:
    python -m src.evaluation.reporting --run-all
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


def _load_predictions() -> dict[str, pd.DataFrame]:
    """Load all prediction parquets from results/predictions/."""
    preds = {}
    for path in sorted(_PREDS_DIR.glob("*.parquet")):
        key = path.stem  # e.g. 'linear_base', 'lstm_regime'
        df = load_parquet(path)
        if df is not None:
            preds[key] = df
    logger.info("Loaded %d prediction files", len(preds))
    return preds


def _returns_from_predictions(
    pred_df: pd.DataFrame,
    equal_weight: bool = True,
) -> pd.Series:
    """Convert predictions DataFrame to portfolio returns.

    Uses sign of prediction as direction, equal-weighted across assets.
    A more sophisticated allocator would use the RL layers -- this function
    provides a simple long/short (long-only here) signal-based portfolio.
    """
    if "prediction" not in pred_df.columns or "target" not in pred_df.columns:
        return pd.Series(dtype=float)

    # Equal-weight across all tickers each date; weight=1/N if pred > 0, else 0
    def _period_return(group):
        signal = (group["prediction"] > 0).astype(float)
        if signal.sum() == 0:
            return 0.0
        weights = signal / signal.sum()
        return float((weights * group["target"]).sum())

    if isinstance(pred_df.index, pd.MultiIndex):
        by_date = pred_df.groupby(level="date")
        daily_returns = by_date.apply(_period_return)
    else:
        daily_returns = pred_df.apply(
            lambda row: row["target"] if row["prediction"] > 0 else 0.0, axis=1
        )

    daily_returns.name = "portfolio_return"
    return daily_returns


def build_comparison_table(
    predictions: dict[str, pd.DataFrame],
    regime_labels: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Compute all metrics for every model-variant and return as a DataFrame.

    Returns
    -------
    pd.DataFrame
        Rows = (model, variant); columns = metric names.
    """
    rows = []
    for model_variant, pred_df in predictions.items():
        parts = model_variant.rsplit("_", 1)
        model = parts[0] if len(parts) == 2 and parts[1] in {"base", "regime"} else model_variant
        variant = parts[1] if len(parts) == 2 and parts[1] in {"base", "regime"} else "base"

        port_returns = _returns_from_predictions(pred_df)
        if len(port_returns) == 0:
            logger.warning("No returns for %s -- skipping", model_variant)
            continue

        metrics = compute_all_metrics(
            portfolio_returns=port_returns,
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


def save_tables(table: pd.DataFrame) -> None:
    _TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _TABLES_DIR / "master_comparison.csv"
    table.to_csv(csv_path)
    logger.info("Master comparison table saved to %s", csv_path)

    md_path = _TABLES_DIR / "master_comparison.md"
    with open(md_path, "w") as f:
        f.write("# Master Model Comparison\n\n")
        # Round floats for readability
        display = table.copy()
        for col in display.select_dtypes(include="number").columns:
            display[col] = display[col].round(4)
        f.write(display.to_markdown())
        f.write("\n")
    logger.info("Markdown table saved to %s", md_path)


def plot_equity_curves(
    predictions: dict[str, pd.DataFrame],
    suffix: str = "base",
) -> None:
    """Plot equity curve overlay for all models with a given variant suffix."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping equity curve plot")
        return

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0

    for model_variant, pred_df in predictions.items():
        if not model_variant.endswith(f"_{suffix}"):
            continue
        port_returns = _returns_from_predictions(pred_df)
        if len(port_returns) == 0:
            continue
        cum = port_returns.cumsum()
        ax.plot(cum.index, cum.values, label=model_variant, alpha=0.8)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(f"Equity Curves — {suffix.title()} Models")
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
        "risk_on": "#2ca02c",   # green
        "neutral": "#1f77b4",   # blue
        "risk_off": "#ff7f0e",  # orange
        "crisis": "#d62728",    # red
    }

    common = spy_returns.index.intersection(regime_labels.index)
    ret = spy_returns.reindex(common)
    labels = regime_labels.reindex(common)
    cum = ret.cumsum()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(cum.index, cum.values, color="black", linewidth=1.5, label="SPY")

    # Shade regime periods
    current_label = None
    start_date = None
    for date, label in zip(labels.index, labels.values):
        if label != current_label:
            if current_label is not None and start_date is not None:
                color = palette.get(str(current_label), "#aaaaaa")
                ax.axvspan(start_date, date, alpha=0.15, color=color)
            current_label = label
            start_date = date
    if current_label is not None and start_date is not None:
        color = palette.get(str(current_label), "#aaaaaa")
        ax.axvspan(start_date, labels.index[-1], alpha=0.15, color=color)

    patches = [
        mpatches.Patch(color=c, alpha=0.3, label=l)
        for l, c in palette.items()
    ]
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
    ax.set_ylabel("Drawdown (log return)")
    ax.legend(loc="lower left", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _FIGURES_DIR / "drawdowns.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Drawdown plot saved: %s", out)


def run_all(rf: float = 0.0) -> pd.DataFrame:
    """Load all cached predictions, compute metrics, save tables and plots."""
    set_all_seeds()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    predictions = _load_predictions()
    if not predictions:
        logger.error("No prediction files found in %s. Run experiments first.", _PREDS_DIR)
        return pd.DataFrame()

    # Load regime labels if available
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

    # Load SPY returns for regime-shaded plot
    spy_path = Path("data/raw/alpaca/SPY.parquet")
    if spy_path.exists() and regime_labels is not None:
        spy_df = load_parquet(spy_path)
        if spy_df is not None and "close" in spy_df.columns:
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
