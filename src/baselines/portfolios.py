"""Passive and rule-based baseline portfolios.

All baselines use EXACTLY the same test dates as the ML prediction models
(passed in as test_dates) to ensure apples-to-apples metric comparison.

Baselines implemented:
  spy_buy_hold     — 100 % SPY, rebalanced never
  equal_weight     — 1/N across all available tickers, rebalanced monthly
  momentum_12_1    — Top-half of universe by 12-month minus 1-month return
  sixty_forty      — 60 % SPY + 40 % TLT, rebalanced monthly

All functions return a pd.Series of daily log returns indexed by test_dates.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ALPACA_DIR = Path("data/raw/alpaca")


def _load_close(ticker: str) -> Optional[pd.Series]:
    path = _ALPACA_DIR / f"{ticker}.parquet"
    if not path.exists():
        logger.warning("Alpaca cache missing for %s", ticker)
        return None
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df["close"]


def _daily_log_ret(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def spy_buy_hold(test_dates: pd.DatetimeIndex) -> pd.Series:
    """100 % SPY — buy at start of test period, hold throughout."""
    spy = _load_close("SPY")
    if spy is None:
        return pd.Series(np.nan, index=test_dates, name="spy_bh")
    ret = _daily_log_ret(spy).reindex(test_dates).fillna(0.0)
    ret.name = "spy_bh"
    return ret


def equal_weight(
    test_dates: pd.DatetimeIndex,
    tickers: list[str],
) -> pd.Series:
    """1/N equal-weight portfolio; signal rebalanced at each fold's first date.

    The signal date (first business day of each calendar month within
    test_dates) is inferred from the fold structure implied by the dates:
    we rebalance whenever the month changes.
    """
    closes = {}
    for t in tickers:
        c = _load_close(t)
        if c is not None:
            closes[t] = c
    if not closes:
        return pd.Series(np.nan, index=test_dates, name="equal_weight")

    close_df = pd.DataFrame(closes)
    daily_ret = np.log(close_df / close_df.shift(1))

    result = daily_ret.reindex(test_dates).mean(axis=1).fillna(0.0)
    result.name = "equal_weight"
    return result


def momentum_12_1(
    test_dates: pd.DatetimeIndex,
    tickers: list[str],
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.Series:
    """Top-half momentum: long tickers with highest (12-month minus 1-month) return.

    Signal is formed at the start of each calendar month (no look-ahead):
    use returns from t-252 to t-21 to rank; hold the top half for the
    following month.
    """
    closes = {}
    for t in tickers:
        c = _load_close(t)
        if c is not None:
            closes[t] = c
    if not closes:
        return pd.Series(np.nan, index=test_dates, name="momentum_12_1")

    close_df = pd.DataFrame(closes).sort_index()
    daily_ret = np.log(close_df / close_df.shift(1))

    # Identify month-start rebalance dates within test_dates
    test_ser = pd.Series(test_dates, index=test_dates)
    rebal_mask = test_ser.index.to_series().diff().dt.days.fillna(99) > 5
    rebal_dates = test_dates[rebal_mask.values]
    if len(rebal_dates) == 0:
        rebal_dates = test_dates[:1]

    port_rets: dict = {}
    current_longs: list[str] = list(closes.keys())

    for i, rd in enumerate(rebal_dates):
        next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None

        # Compute momentum signal as of rd (using data up to rd, no look-ahead)
        loc = close_df.index.searchsorted(rd, side="right")
        if loc >= lookback_days + skip_days:
            past = close_df.iloc[loc - lookback_days : loc - skip_days]
            recent = close_df.iloc[loc - skip_days : loc]
            if len(past) > 0 and len(recent) > 0:
                mom = np.log(past.iloc[-1] / past.iloc[0]) - np.log(recent.iloc[-1] / recent.iloc[0])
                mom = mom.dropna()
                med = mom.median()
                current_longs = mom[mom >= med].index.tolist()

        # Apply signal for each date in this fold
        fold_dates = test_dates[test_dates >= rd]
        if next_rd is not None:
            fold_dates = fold_dates[fold_dates < next_rd]

        for d in fold_dates:
            avail = [t for t in current_longs if t in daily_ret.columns and d in daily_ret.index]
            if avail:
                port_rets[d] = daily_ret.loc[d, avail].mean()
            else:
                port_rets[d] = 0.0

    result = pd.Series(port_rets).reindex(test_dates).fillna(0.0)
    result.name = "momentum_12_1"
    return result


def sixty_forty(
    test_dates: pd.DatetimeIndex,
    spy_weight: float = 0.60,
    bond_ticker: str = "TLT",
) -> pd.Series:
    """60/40 portfolio: spy_weight in SPY, (1-spy_weight) in bond_ticker.

    Static weights, no rebalancing drift correction (constant proportions).
    """
    spy = _load_close("SPY")
    bond = _load_close(bond_ticker)

    if spy is None or bond is None:
        logger.warning("60/40: missing SPY or %s — returning zeros", bond_ticker)
        return pd.Series(0.0, index=test_dates, name="sixty_forty")

    spy_ret = _daily_log_ret(spy).reindex(test_dates).fillna(0.0)
    bond_ret = _daily_log_ret(bond).reindex(test_dates).fillna(0.0)

    result = spy_weight * spy_ret + (1 - spy_weight) * bond_ret
    result.name = "sixty_forty"
    return result
