"""FRED data loader.

Pulls macro series via fredapi and caches to data/raw/fred/{series_id}.parquet.
Never silently swaps series IDs -- raises if a series is empty or unavailable.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.io import cache_hit, load_parquet, save_parquet

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/raw/fred")


def _get_fred():
    """Return a fredapi.Fred instance using FRED_API_KEY env var."""
    try:
        from fredapi import Fred
    except ImportError as e:
        raise ImportError("fredapi is required: pip install fredapi") from e

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY must be set as an environment variable.")
    return Fred(api_key=api_key)


def _cache_path(series_id: str) -> Path:
    return _CACHE_DIR / f"{series_id}.parquet"


def load_series(
    series_id: str,
    start: str = "2014-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Pull a FRED series and return as a pd.Series indexed by date.

    Parameters
    ----------
    series_id:
        FRED series identifier (e.g. 'DGS10').
    start:
        Start date for the pull (2 years before model start for lagged features).
    end:
        End date (defaults to today).
    force_refresh:
        Re-pull even if a cache file exists.

    Returns
    -------
    pd.Series
        Values indexed by pd.DatetimeIndex, named with series_id.

    Raises
    ------
    RuntimeError
        If FRED returns an empty series -- never fall back silently.
    """
    path = _cache_path(series_id)

    if not force_refresh and cache_hit(path):
        logger.info("Cache hit for FRED:%s", series_id)
        df = load_parquet(path)
        return df["value"]

    fred = _get_fred()
    logger.info("Pulling FRED series %s (start=%s)", series_id, start)

    try:
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
    except Exception as exc:
        raise RuntimeError(
            f"FRED API error for series '{series_id}': {exc}. "
            "Verify the series ID is correct and your FRED_API_KEY is valid."
        ) from exc

    if s is None or len(s) == 0:
        raise RuntimeError(
            f"FRED returned an empty series for '{series_id}'. "
            "Do not substitute a different series ID -- surface this error."
        )

    s.index = pd.DatetimeIndex(s.index).normalize()
    s.index.name = "date"
    s.name = series_id

    df = s.to_frame(name="value")
    save_parquet(df, path)
    logger.info("Cached %d observations for FRED:%s", len(s), series_id)
    return s


def load_all_series(
    series_ids: list[str],
    start: str = "2014-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull multiple FRED series and return as a wide DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns = series IDs, index = date, aligned via outer join and
        forward-filled to handle different release frequencies.

    Raises
    ------
    RuntimeError
        If any series fails -- all must succeed for a valid feature panel.
    """
    frames: dict[str, pd.Series] = {}
    errors: list[str] = []

    for sid in series_ids:
        try:
            frames[sid] = load_series(sid, start=start, end=end, force_refresh=force_refresh)
        except Exception as exc:
            logger.error("FRED load failed for %s: %s", sid, exc)
            errors.append(f"{sid}: {exc}")

    if errors:
        raise RuntimeError(
            f"Failed to load {len(errors)} FRED series:\n" + "\n".join(errors)
        )

    df = pd.DataFrame(frames)
    df = df.sort_index().ffill()
    logger.info("Loaded %d FRED series, %d dates", len(frames), len(df))
    return df
