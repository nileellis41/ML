"""Feature engineering for the ETF research pipeline.

Constructs the universal macro block and sector-specific feature blocks
for each ETF, then applies return/vol transformations and z-scoring.
All transformations are strictly backward-looking (no look-ahead).
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.fred_loader import load_all_series
from src.utils.io import load_parquet

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path("config/features.yaml")
_ASSETS_PATH = Path("config/assets.yaml")


def _load_config() -> tuple[dict, dict]:
    with open(_CONFIG_PATH) as f:
        feat_cfg = yaml.safe_load(f)
    with open(_ASSETS_PATH) as f:
        asset_cfg = yaml.safe_load(f)
    return feat_cfg, asset_cfg


def _compute_returns(prices: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Compute log returns over multiple look-back windows."""
    frames = {}
    for w in windows:
        frames[f"ret_{w}d"] = np.log(prices / prices.shift(w))
    return pd.DataFrame(frames, index=prices.index)


def _compute_realized_vol(returns_1d: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Annualised realised volatility over rolling windows."""
    frames = {}
    for w in windows:
        frames[f"rvol_{w}d"] = returns_1d.rolling(w).std() * np.sqrt(252)
    return pd.DataFrame(frames, index=returns_1d.index)


def _zscore_series(s: pd.Series, window: int = 252) -> pd.Series:
    """Rolling z-score to remove level non-stationarity from macro series."""
    mu = s.rolling(window, min_periods=60).mean()
    sigma = s.rolling(window, min_periods=60).std()
    return (s - mu) / sigma.replace(0, np.nan)


def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Replace values with cross-sectional percentile ranks per date row."""
    return df.rank(axis=1, pct=True)


def build_price_features(
    ticker: str,
    price_df: pd.DataFrame,
    return_windows: list[int],
    vol_windows: list[int],
) -> pd.DataFrame:
    """Build return and volatility features from OHLCV data for one ticker."""
    close = price_df["close"]
    ret_feats = _compute_returns(close, return_windows)
    vol_feats = _compute_realized_vol(ret_feats["ret_1d"], vol_windows)
    return pd.concat([ret_feats, vol_feats], axis=1).add_prefix(f"{ticker}_")


def build_universal_macro_features(
    universal_ids: list[str],
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Pull and transform universal macro FRED series.

    Applies rolling z-scoring to all levels; adds term-spread if DGS10/DGS2
    are both present.
    """
    raw = load_all_series(universal_ids, start=start)

    transformed = {}
    for col in raw.columns:
        s = raw[col].dropna()
        transformed[f"macro_{col}"] = _zscore_series(raw[col])
        # Also include 1-period delta (change in macro level)
        transformed[f"macro_{col}_delta"] = raw[col].diff()

    df = pd.DataFrame(transformed)

    # Derived: term spread if both yields available
    if "DGS10" in raw.columns and "DGS2" in raw.columns:
        spread = raw["DGS10"] - raw["DGS2"]
        df["macro_term_spread"] = _zscore_series(spread)
        df["macro_term_spread_delta"] = spread.diff()

    return df


def build_sector_features(
    ticker: str,
    sector_fred_ids: list[str],
    cross_asset_tickers: list[str],
    alpaca_cache_dir: Path,
    start: str = "2014-01-01",
    return_windows: Optional[list[int]] = None,
    vol_windows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Build sector-specific feature block for one ETF.

    Combines FRED pulls with cross-asset price features (e.g., SOXX for XLK).
    """
    return_windows = return_windows or [20, 60]
    vol_windows = vol_windows or [60]
    frames = []

    # Sector FRED series
    if sector_fred_ids:
        raw = load_all_series(sector_fred_ids, start=start)
        transformed = {}
        for col in raw.columns:
            transformed[f"sector_{col}"] = _zscore_series(raw[col])
            transformed[f"sector_{col}_delta"] = raw[col].diff()
        frames.append(pd.DataFrame(transformed))

    # Cross-asset price features (e.g., SOXX 20d return as feature for XLK)
    for ca_ticker in cross_asset_tickers:
        path = alpaca_cache_dir / f"{ca_ticker}.parquet"
        ca_df = load_parquet(path)
        if ca_df is None:
            logger.warning(
                "Cross-asset %s not cached for %s; skipping feature.", ca_ticker, ticker
            )
            continue
        close = ca_df["close"]
        ret_feats = _compute_returns(close, return_windows).add_prefix(f"ca_{ca_ticker}_")
        vol_feats = _compute_realized_vol(
            np.log(close / close.shift(1)), vol_windows
        ).add_prefix(f"ca_{ca_ticker}_")
        frames.append(pd.concat([ret_feats, vol_feats], axis=1))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined.index.name = "date"
    return combined


def build_feature_matrix(
    ticker: str,
    price_df: pd.DataFrame,
    feat_config: dict,
    alpaca_cache_dir: Path = Path("data/raw/alpaca"),
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Assemble the full feature matrix for one ETF.

    Returns
    -------
    pd.DataFrame
        All features (price-derived + universal macro + sector-specific)
        aligned to the ETF's trading calendar.  No future data.
    """
    fe_cfg = feat_config.get("feature_engineering", {})
    return_windows = fe_cfg.get("return_windows", [1, 5, 20, 60])
    vol_windows = fe_cfg.get("vol_windows", [20, 60])
    universal_ids = feat_config.get("universal_macro", {}).get("fred", [])
    sector_block = feat_config.get("sector_features", {}).get(ticker, {})
    sector_fred_ids = sector_block.get("fred", [])
    cross_asset = sector_block.get("cross_asset", [])

    # 1. Price features for this ETF
    price_feats = build_price_features(ticker, price_df, return_windows, vol_windows)

    # 2. Universal macro block
    macro_feats = build_universal_macro_features(universal_ids, start=start)

    # 3. Sector-specific block
    sector_feats = build_sector_features(
        ticker, sector_fred_ids, cross_asset, alpaca_cache_dir,
        start=start, return_windows=[20, 60], vol_windows=[60],
    )

    # Align everything to the ETF's date index (forward-fill macro on non-trading days)
    idx = price_feats.index
    macro_feats = macro_feats.reindex(idx, method="ffill")
    sector_feats = sector_feats.reindex(idx, method="ffill") if not sector_feats.empty else pd.DataFrame(index=idx)

    combined = pd.concat([price_feats, macro_feats, sector_feats], axis=1)
    combined.index.name = "date"

    n_before = len(combined)
    combined = combined.dropna(how="all")
    logger.info(
        "%s: feature matrix shape %s (dropped %d all-NaN rows)",
        ticker, combined.shape, n_before - len(combined),
    )
    return combined


def add_cross_sectional_ranks(
    feature_dict: dict[str, pd.DataFrame],
    column_suffix: str = "_ret_20d",
) -> dict[str, pd.DataFrame]:
    """Add cross-sectional rank features across the ETF universe per date.

    Ranks each ETF's 20d return relative to its peers and appends as a new
    feature column.  Only uses tickers present in feature_dict.
    """
    # Collect the target column across all tickers
    rank_series: dict[str, pd.Series] = {}
    for ticker, df in feature_dict.items():
        col = f"{ticker}{column_suffix}"
        if col in df.columns:
            rank_series[ticker] = df[col]

    if len(rank_series) < 2:
        return feature_dict

    panel = pd.DataFrame(rank_series)
    ranks = panel.rank(axis=1, pct=True)  # cross-sectional pct rank per date

    for ticker in feature_dict:
        if ticker in ranks.columns:
            feature_dict[ticker]["xsrank_ret20d"] = ranks[ticker]

    return feature_dict
