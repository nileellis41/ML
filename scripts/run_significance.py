"""Run Diebold-Mariano significance tests and deflated Sharpe for all strategies.

Produces:
  results/tables/significance_vs_baselines.csv  (32 rows, 8 ML × 4 baselines)
  results/tables/significance_summary.md

Run from project root:
    python scripts/run_significance.py

Exit codes:
    0  — no ML model survives Bonferroni at p < 0.05
    1  — at least one ML model survives Bonferroni
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
import numpy as np
import pandas as pd

from src.evaluation.reporting import _load_predictions, _returns_from_predictions
from src.baselines.portfolios import (
    equal_weight, momentum_12_1, sixty_forty, spy_buy_hold,
)
from src.evaluation.significance import run_significance_tests, format_significance_summary
from src.evaluation.deflated_sharpe import compute_dsr_for_returns
from src.utils.seeds import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "results" / "tables"
N_TRIALS = 8  # number of ML configurations evaluated


def main() -> int:
    set_all_seeds()

    # ---- Load ML portfolio returns ----
    predictions = _load_predictions()
    if not predictions:
        logger.error("No prediction files found. Run experiments first.")
        return 1

    # Skip determinism test copies
    predictions = {k: v for k, v in predictions.items()
                   if "run1" not in k and "run2" not in k}

    ml_returns: dict[str, pd.Series] = {}
    for key, pred_df in predictions.items():
        ret = _returns_from_predictions(pred_df)
        if len(ret) > 0:
            ml_returns[key] = ret
            logger.info("Loaded ML returns: %s  (%d days)", key, len(ret))

    if not ml_returns:
        logger.error("Could not compute portfolio returns from any prediction file.")
        return 1

    # ---- Build baseline returns on the same test dates ----
    first_pred = next(iter(predictions.values()))
    test_dates = pd.DatetimeIndex(
        sorted(first_pred.index.get_level_values("date").unique())
    )
    tickers = first_pred.index.get_level_values("ticker").unique().tolist()

    baseline_returns: dict[str, pd.Series] = {
        "spy_bh":        spy_buy_hold(test_dates).fillna(0.0),
        "equal_weight":  equal_weight(test_dates, tickers).fillna(0.0),
        "momentum_12_1": momentum_12_1(test_dates, tickers).fillna(0.0),
        "sixty_forty":   sixty_forty(test_dates).fillna(0.0),
    }

    # ---- Run DM tests ----
    logger.info("Running Diebold-Mariano tests (%d ML × %d baselines) ...",
                len(ml_returns), len(baseline_returns))
    sig_df = run_significance_tests(ml_returns, baseline_returns)

    # ---- Deflated Sharpe for all strategies ----
    logger.info("Computing Deflated Sharpe Ratios (n_trials=%d) ...", N_TRIALS)
    dsr_rows = []
    all_strats = {**ml_returns, **baseline_returns}
    for name, ret in sorted(all_strats.items()):
        dsr = compute_dsr_for_returns(ret, n_trials=N_TRIALS)
        dsr_rows.append({"strategy": name, "deflated_sharpe_pvalue": dsr})
        logger.info("  %-30s  DSR p-value = %.4f", name, dsr if not np.isnan(dsr) else float("nan"))

    dsr_df = pd.DataFrame(dsr_rows)
    dsr_path = OUT_DIR / "deflated_sharpe.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dsr_df.to_csv(dsr_path, index=False)
    logger.info("Deflated Sharpe saved: %s", dsr_path)

    # ---- Save significance results ----
    sig_path = OUT_DIR / "significance_vs_baselines.csv"
    sig_df.to_csv(sig_path, index=False)
    logger.info("Significance CSV saved: %s  (%d rows)", sig_path, len(sig_df))

    summary_text = format_significance_summary(sig_df)
    md_path = OUT_DIR / "significance_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info("Significance summary saved: %s", md_path)

    # ---- Print summary to stdout ----
    # Force UTF-8 on Windows stdout to handle any Unicode in the summary
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("\n" + summary_text)
    sys.stdout.flush()

    n_sig = int(sig_df["significant_at_05_after_bonferroni"].sum())
    return 1 if n_sig > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
