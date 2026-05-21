"""Alpaca OHLCV data loader.

Pulls daily adjusted bars via alpaca-py StockHistoricalDataClient.
Caches to data/raw/alpaca/{ticker}.parquet.
Raises immediately if a ticker is unavailable -- no silent fallbacks.
"""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.utils.io import cache_hit, load_parquet, save_parquet

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/raw/alpaca")


def _get_client() -> StockHistoricalDataClient:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set as environment variables."
        )
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _cache_path(ticker: str) -> Path:
    return _CACHE_DIR / f"{ticker}.parquet"


def load_ticker(
    ticker: str,
    start: str = "2016-01-01",
    end: Optional[str] = None,
    feed: str = "iex",
    adjustment: str = "all",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV bars for *ticker* as a DataFrame indexed by date.

    Parameters
    ----------
    ticker:
        Alpaca-recognised ticker symbol.
    start:
        ISO date string for the start of the historical window.
    end:
        ISO date string for end (defaults to today).
    feed:
        Alpaca data feed -- 'iex' for free tier, 'sip' if subscribed.
    adjustment:
        Price adjustment type; 'all' applies split + dividend adjustments.
    force_refresh:
        Re-pull from Alpaca even if a cache file already exists.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, vwap, trade_count.
        Index: pd.DatetimeIndex (date, UTC-normalised).

    Raises
    ------
    RuntimeError
        If Alpaca returns no data for the ticker -- never falls back silently.
    """
    path = _cache_path(ticker)

    if not force_refresh and cache_hit(path):
        logger.info("Cache hit for %s -- loading from %s", ticker, path)
        return load_parquet(path)

    end_dt = end or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    logger.info("Pulling %s from Alpaca (%s to %s, feed=%s)", ticker, start, end_dt, feed)

    client = _get_client()
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start,
        end=end_dt,
        adjustment=adjustment,
        feed=feed,
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if df is None or df.empty:
        raise RuntimeError(
            f"Alpaca returned no data for '{ticker}' ({start} to {end_dt}, feed={feed}). "
            "Verify the ticker is available on the configured feed and within the date range."
        )

    # Normalise multi-index (symbol, timestamp) -> flat date index
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]

    save_parquet(df, path)
    logger.info("Cached %d rows for %s to %s", len(df), ticker, path)
    return df


def load_universe(
    tickers: list[str],
    start: str = "2016-01-01",
    end: Optional[str] = None,
    feed: str = "iex",
    adjustment: str = "all",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Pull OHLCV for all tickers in the universe; fail loudly on any error.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ticker -> daily OHLCV DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for ticker in tickers:
        try:
            results[ticker] = load_ticker(
                ticker, start=start, end=end,
                feed=feed, adjustment=adjustment,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            logger.error("Failed to load %s: %s", ticker, exc)
            errors.append(f"{ticker}: {exc}")

    if errors:
        raise RuntimeError(
            f"Failed to load {len(errors)} ticker(s):\n" + "\n".join(errors)
        )

    logger.info("Loaded %d tickers successfully", len(results))
    return results


def smoke_test(ticker: str = "SPY", n_rows: int = 20) -> pd.DataFrame:
    """Quick smoke test: pull the last *n_rows* rows for *ticker*."""
    from datetime import timedelta
    end = datetime.now(tz=timezone.utc)
    start = (end - timedelta(days=60)).strftime("%Y-%m-%d")
    df = load_ticker(ticker, start=start, force_refresh=True)
    logger.info("Smoke test OK: %s shape=%s\n%s", ticker, df.shape, df.tail(n_rows))
    return df.tail(n_rows)
