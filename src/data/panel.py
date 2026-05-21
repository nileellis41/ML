"""Multi-asset feature panel builder.

Assembles a panel DataFrame indexed by (date, ticker) with all features
and a forward return target for each ETF in the universe.
Saves to data/processed/panel.parquet.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.alpaca_loader import load_universe
from src.data.features import add_cross_sectional_ranks, build_feature_matrix
from src.utils.io import cache_hit, load_parquet, save_parquet

logger = logging.getLogger(__name__)

_PANEL_PATH = Path("data/processed/panel.parquet")
_CONFIG_ASSETS = Path("config/assets.yaml")
_CONFIG_FEATURES = Path("config/features.yaml")


def _forward_return(close: pd.Series, horizon_days: int = 21) -> pd.Series:
    """21-trading-day forward log return (approx 1 calendar month).

    Aligned so index date t has the return from t+1 to t+horizon_days.
    This is the regression target -- strictly forward-looking.
    """
    return np.log(close.shift(-horizon_days) / close)


def build_panel(
    start: str = "2016-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
    horizon_days: int = 21,
) -> pd.DataFrame:
    """Build and cache the aligned multi-asset feature panel.

    Parameters
    ----------
    start:
        Data pull start (extra history needed for feature computation).
    end:
        Data pull end.
    force_refresh:
        Re-build even if panel.parquet exists.
    horizon_days:
        Trading days for the forward-return target (default 21 ~ 1 month).

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, ticker), columns = features + 'target'.
    """
    if not force_refresh and cache_hit(_PANEL_PATH):
        logger.info("Loading cached panel from %s", _PANEL_PATH)
        return load_parquet(_PANEL_PATH)

    with open(_CONFIG_ASSETS) as f:
        asset_cfg = yaml.safe_load(f)
    with open(_CONFIG_FEATURES) as f:
        feat_cfg = yaml.safe_load(f)

    # Target tickers: sector ETFs + indices + commodities + bonds
    target_tickers = (
        asset_cfg["sector_etfs"]
        + asset_cfg["indices"]
        + asset_cfg["commodities"]
        + asset_cfg["bond_etfs"]
    )
    all_tickers = asset_cfg["all_tickers"]  # includes cross-asset feature tickers

    logger.info("Pulling OHLCV for %d tickers", len(all_tickers))
    ohlcv = load_universe(
        all_tickers, start="2014-01-01", end=end,
        feed=asset_cfg["data"]["feed"],
        adjustment=asset_cfg["data"]["adjustment"],
    )

    alpaca_cache = Path(asset_cfg["data"]["cache_dir"])
    feature_dict: dict[str, pd.DataFrame] = {}

    for ticker in target_tickers:
        if ticker not in ohlcv:
            logger.warning("Ticker %s not in OHLCV results -- skipping", ticker)
            continue
        logger.info("Building features for %s", ticker)
        df = build_feature_matrix(
            ticker, ohlcv[ticker], feat_cfg,
            alpaca_cache_dir=alpaca_cache, start="2014-01-01",
        )
        # Add forward return target
        df["target"] = _forward_return(ohlcv[ticker]["close"], horizon_days)
        feature_dict[ticker] = df

    # Add cross-sectional rank features
    feature_dict = add_cross_sectional_ranks(feature_dict)

    # Stack into (date, ticker) MultiIndex panel
    frames = []
    for ticker, df in feature_dict.items():
        df = df.copy()
        df["ticker"] = ticker
        df.index.name = "date"
        frames.append(df.reset_index().set_index(["date", "ticker"]))

    panel = pd.concat(frames, axis=0).sort_index()

    # Trim to the requested start date (feature history requires extra look-back)
    panel = panel[panel.index.get_level_values("date") >= start]

    save_parquet(panel, _PANEL_PATH)
    logger.info("Panel saved: %s rows, %s columns", len(panel), len(panel.columns))
    return panel
