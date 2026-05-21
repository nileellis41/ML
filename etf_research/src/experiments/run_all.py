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
from src.utils.io import load_parquet, save_parquet
from src.utils.seeds import set_all_seeds

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


def _load_regime_probs() -> Optional[pd.DataFrame]:
    path = _REGIMES_DIR / "hmm_probs.parquet"
    if not path.exists():
        logger.warning("HMM regime probs not found at %s -- regime variants will be skipped.", path)
        return None
    return load_parquet(path)


def _get_feature_cols(panel: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    exclude = exclude or ["target"]
    return [c for c in panel.columns if c not in exclude]


def _append_regime_features(
    panel: pd.DataFrame,
    regime_df: pd.DataFrame,
    n_states: int = 4,
) -> pd.DataFrame:
    """Append regime state probability columns to the panel."""
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

    return pd.concat([panel, regime_reindexed], axis=1)


def run_prediction_models(
    panel: pd.DataFrame,
    folds: list,
    exp_cfg: dict,
    models_to_run: list[str],
    regime_df: Optional[pd.DataFrame] = None,
) -> None:
    """Walk-forward all requested prediction models and save results."""
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
                if k not in ("seq_len",)  # handled separately
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
        logger.info("Running %s_base...", model_name)
        model = model_cls(**kwargs)
        results = run_walk_forward(model, panel, folds, feature_cols)
        if not results.empty:
            save_parquet(results, _PREDS_DIR / f"{model_name}_base.parquet")
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
) -> None:
    set_all_seeds()
    exp_cfg, asset_cfg = _load_configs()

    # Step 1: FRED validation
    if not skip_validation:
        logger.info("=== Step 1: FRED Series Validation ===")
        validate_all()

    # Step 2: Build panel
    logger.info("=== Step 2: Building Feature Panel ===")
    panel = build_panel(force_refresh=refit)

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

    # Step 4: Load regime probs (if available)
    regime_df = _load_regime_probs()

    # Step 5: Run models
    logger.info("=== Step 5: Running Prediction Models ===")
    _models = models_to_run or ["linear", "logistic", "pca", "lstm"]
    run_prediction_models(panel, folds, exp_cfg, _models, regime_df=regime_df)

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
    args = parser.parse_args()
    main(refit=args.refit, models_to_run=args.models, skip_validation=args.skip_validation)
