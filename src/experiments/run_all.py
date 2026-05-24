"""Master experiment runner.

Executes the full walk-forward pipeline end-to-end:
  1. Validate FRED series
  2. Build feature panel
  3. Fit HMM regime model
  4. Run walk-forward for all prediction models (base and regime variants)
  5. Run portfolio allocators
  6. Save prediction parquets to results/predictions/

Usage:
    python -m src.experiments.run_all --refit
    python -m src.experiments.run_all --refit --models linear logistic
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.panel import build_panel
from src.data.validate_fred import validate_all
from src.evaluation.walk_forward import assert_no_leakage, generate_folds, run_walk_forward
from src.models.linear import LinearReturnModel, LogisticDirectionModel
from src.models.pca_factor import PCAFactorModel
from src.models.var import VARReturnModel
from src.models.lstm import LSTMReturnModel
from src.regimes.hmm import fit_hmm_rolling
from src.utils.io import load_parquet, save_parquet
from src.utils.seeds import set_all_seeds

_ALPACA_CACHE = Path("data/raw/alpaca")
_FRED_CACHE = Path("data/raw/fred")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

_PREDS_DIR = Path("results/predictions")
_REGIMES_DIR = Path("data/regimes")
_CONFIG_EXP = Path("config/experiments.yaml")
_CONFIG_ASSETS = Path("config/assets.yaml")


def _load_configs() -> tuple[dict, dict]:
    with open(_CONFIG_EXP) as f:
        exp_cfg = yaml.safe_load(f)
    with open(_CONFIG_ASSETS) as f:
        asset_cfg = yaml.safe_load(f)
    return exp_cfg, asset_cfg


def run_hmm(exp_cfg: dict) -> Optional[pd.DataFrame]:
    """Load inputs and run rolling-walk-forward HMM regime detection.

    Returns the regime probability DataFrame (also written to disk).
    """
    hmm_cfg = exp_cfg.get("regime", {}).get("hmm", {})

    def _load_fred(series_id: str) -> pd.Series:
        path = _FRED_CACHE / f"{series_id}.parquet"
        df = load_parquet(path)
        if df is None:
            raise RuntimeError(f"FRED cache missing for {series_id}")
        return df.squeeze()

    # Use NASDAQCOM (FRED, 2014–present) as the market-return input.
    # SPY from Alpaca IEX has a ~634-day gap before 2020-07, which pushes the
    # effective HMM feature window to 2020-10 and leaves only 7 months of OOS
    # labels — far too few to cover the model training period.
    # NASDAQCOM gives an OOS window from ~2019-05 onward, covering all folds.
    market_close = _load_fred("NASDAQCOM")

    vix = _load_fred("VIXCLS")
    # BAMLH0A0HYM2 (HY OAS) is restricted by FRED/ICE licensing to ~3 years;
    # omit it so the HMM uses the full 12-year NASDAQCOM/VIX/term-spread history.
    dgs10 = _load_fred("DGS10")
    dgs2 = _load_fred("DGS2")
    term_spread = dgs10 - dgs2

    result = fit_hmm_rolling(
        market_close=market_close,
        vix=vix,
        hy_oas=None,
        term_spread=term_spread,
        initial_train_years=hmm_cfg.get("initial_train_years", 5),
        n_states=hmm_cfg.get("n_states", 4),
        n_iter=hmm_cfg.get("n_iter", 100),
        covariance_type=hmm_cfg.get("covariance_type", "full"),
        version=hmm_cfg.get("version", "v1"),
    )
    logger.info(
        "HMM complete: %d out-of-sample dates, label distribution:\n%s",
        len(result),
        result["predicted_label"].value_counts().to_string(),
    )
    return result


def _load_regime_probs() -> Optional[pd.DataFrame]:
    path = _REGIMES_DIR / "hmm_probs.parquet"
    if not path.exists():
        logger.warning("HMM regime probs not found at %s -- regime variants will be skipped.", path)
        return None
    return load_parquet(path)


def _get_feature_cols(panel: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    exclude = exclude or ["target"]
    return [c for c in panel.columns if c not in exclude]


_REGIME_INTERACTION_BASE_COLS = [
    "own_ret_1d", "own_ret_5d", "own_ret_20d", "own_ret_60d",
    "own_rvol_20d", "own_rvol_60d",
]


def _append_regime_features(
    panel: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_states: int = 4,
) -> pd.DataFrame:
    """Append regime state probability columns AND interaction features to the panel.

    Interaction features are state_j × own_ret/vol_Xd for each state j and
    each price-feature column.  These have genuine cross-sectional variance
    (unlike raw state probs which are constant across all tickers on a date)
    and allow cross-sectional models to learn regime-conditioned rankings.
    """
    state_cols = [f"state_{i}" for i in range(n_states)]
    available = [c for c in state_cols if c in regime_df.columns]
    if not available:
        logger.warning("No state columns found in regime DataFrame.")
        return panel

    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values("date")
        regime_reindexed = regime_df[available].reindex(dates).fillna(0.25)
        regime_reindexed.index = panel.index
    else:
        regime_reindexed = regime_df[available].reindex(panel.index, method="ffill").fillna(0.25)

    panel_with_states = pd.concat([panel, regime_reindexed], axis=1)

    # Build interaction features: state_j × own_feature for cross-sectional variance
    interaction_frames = []
    interaction_cols = [c for c in _REGIME_INTERACTION_BASE_COLS if c in panel.columns]
    for state_col in available:
        for feat_col in interaction_cols:
            col_name = f"{state_col}_x_{feat_col}"
            interaction_frames.append(
                (panel_with_states[state_col] * panel_with_states[feat_col]).rename(col_name)
            )

    if interaction_frames:
        interactions = pd.concat(interaction_frames, axis=1)
        return pd.concat([panel_with_states, interactions], axis=1)

    return panel_with_states


def run_prediction_models(
    panel: pd.DataFrame,
    folds: list,
    exp_cfg: dict,
    models_to_run: list[str],
    regime_df: Optional[pd.DataFrame] = None,
    skip_existing_base: bool = False,
) -> None:
    """Walk-forward all requested prediction models and save results.

    Parameters
    ----------
    skip_existing_base:
        If True, skip re-running the base variant when the parquet already
        exists on disk.  Useful for --regime-only runs.
    """
    _PREDS_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = _get_feature_cols(panel)
    regime_feature_cols = None

    if regime_df is not None:
        panel_regime = _append_regime_features(panel, regime_df)
        regime_feature_cols = _get_feature_cols(panel_regime)

    model_configs = {
        "linear": {
            "model_cls": LinearReturnModel,
            "kwargs": exp_cfg.get("models", {}).get("linear", {}),
        },
        "logistic": {
            "model_cls": LogisticDirectionModel,
            "kwargs": exp_cfg.get("models", {}).get("logistic", {}),
        },
        "pca": {
            "model_cls": PCAFactorModel,
            "kwargs": exp_cfg.get("models", {}).get("pca_factor", {}),
        },
        "lstm": {
            "model_cls": LSTMReturnModel,
            "kwargs": {
                k: v for k, v in exp_cfg.get("models", {}).get("lstm", {}).items()
                if k not in ("seq_len",)
            },
        },
    }

    for model_name in models_to_run:
        if model_name not in model_configs:
            logger.warning("Unknown model '%s' -- skipping.", model_name)
            continue

        cfg = model_configs[model_name]
        model_cls = cfg["model_cls"]
        kwargs = cfg["kwargs"]

        # --- Base variant ---
        base_path = _PREDS_DIR / f"{model_name}_base.parquet"
        if skip_existing_base and base_path.exists():
            logger.info("Skipping %s_base (already exists, --regime-only mode).", model_name)
        else:
            logger.info("Running %s_base...", model_name)
            model = model_cls(**kwargs)
            results = run_walk_forward(model, panel, folds, feature_cols)
            if not results.empty:
                save_parquet(results, base_path)
                logger.info("Saved %s_base: %d rows", model_name, len(results))

        # --- Regime variant ---
        if regime_df is not None and regime_feature_cols is not None:
            logger.info("Running %s_regime...", model_name)
            model_r = model_cls(**kwargs)
            results_r = run_walk_forward(model_r, panel_regime, folds, regime_feature_cols)
            if not results_r.empty:
                save_parquet(results_r, _PREDS_DIR / f"{model_name}_regime.parquet")
                logger.info("Saved %s_regime: %d rows", model_name, len(results_r))


def main(
    refit: bool = True,
    models_to_run: Optional[list[str]] = None,
    skip_validation: bool = False,
    run_hmm_step: bool = False,
    regime_only: bool = False,
) -> None:
    set_all_seeds()
    exp_cfg, asset_cfg = _load_configs()

    # Step 1: FRED validation
    if not skip_validation:
        logger.info("=== Step 1: FRED Series Validation ===")
        validate_all()

    # Step 2: Build panel
    logger.info("=== Step 2: Building Feature Panel ===")
    panel = build_panel(force_refresh=refit and not regime_only)

    # Step 3: Generate walk-forward folds
    logger.info("=== Step 3: Walk-Forward Folds ===")
    wf_cfg = exp_cfg.get("walk_forward", {})
    dates = panel.index.get_level_values("date").unique()
    folds = generate_folds(
        dates,
        initial_train_years=wf_cfg.get("initial_train_years", 5),
        test_months=wf_cfg.get("test_months", 1),
        refit_cadence_months=wf_cfg.get("refit_cadence", {}).get("slow_models", 12),
    )
    assert_no_leakage(folds, panel)
    logger.info("Leakage check PASSED. %d folds generated.", len(folds))

    # Step 4: HMM regime detection
    if run_hmm_step:
        logger.info("=== Step 4: Running HMM Regime Detection ===")
        run_hmm(exp_cfg)

    regime_df = _load_regime_probs()

    # Step 5: Run models
    logger.info("=== Step 5: Running Prediction Models ===")
    _models = models_to_run or ["linear", "logistic", "pca", "lstm"]
    run_prediction_models(
        panel, folds, exp_cfg, _models,
        regime_df=regime_df,
        skip_existing_base=regime_only,
    )

    logger.info("=== Pipeline Complete ===")
    logger.info("Run: python -m src.evaluation.reporting --run-all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF Research Pipeline")
    parser.add_argument("--refit", action="store_true", default=True,
                        help="Re-run all models (ignore cached predictions)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models to run: linear logistic pca lstm cnn var")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip FRED series validation (not recommended)")
    parser.add_argument("--run-hmm", action="store_true",
                        help="Run HMM regime detection before model variants")
    parser.add_argument("--regime-only", action="store_true",
                        help="Skip re-running base variants; only produce regime parquets")
    args = parser.parse_args()
    main(
        refit=args.refit,
        models_to_run=args.models,
        skip_validation=args.skip_validation,
        run_hmm_step=args.run_hmm,
        regime_only=args.regime_only,
    )
