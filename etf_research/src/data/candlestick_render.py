"""Candlestick image rendering pipeline.

Renders OHLCV bars as grayscale images for the CNN model.
Inspired by Jiang/Kelly/Xiu (2023) -- the CNN learns visual patterns
without hand-coded technical indicators.

Image spec
----------
- 64 x 60 pixels (height x width), one column per trading day
- Channel 0: OHLC candle (wick + body on black background, white fills)
- Channel 1: volume bar row (optional, stacked below candle area)
- All prices min-max scaled within each window (pattern learning, not level)
- Cached as compressed .npy arrays in data/processed/candles/{ticker}/{date}.npy
- No look-ahead: the image for date t uses only data through t's close
"""
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IMG_HEIGHT = 64
_IMG_WIDTH = 60   # trading days per window
_CANDLE_BODY_WIDTH = 3
_WICK_WIDTH = 1
_CACHE_DIR = Path("data/processed/candles")
_SAMPLE_DIR = Path("results/figures/candle_samples")


def _render_single_image(
    ohlcv_window: pd.DataFrame,
    include_volume: bool = True,
) -> np.ndarray:
    """Render one 60-day OHLCV window as a (1 or 2, 64, 60) uint8 array.

    Parameters
    ----------
    ohlcv_window:
        DataFrame with columns [open, high, low, close, volume], exactly
        60 rows (trading days), sorted ascending by date.
    include_volume:
        If True, return 2-channel array with volume row as channel 1.

    Returns
    -------
    np.ndarray
        Shape (C, H, W) = (2 if include_volume else 1, 64, 60), dtype uint8.
        White (255) on black (0) background.
    """
    assert len(ohlcv_window) == _IMG_WIDTH, (
        f"Expected {_IMG_WIDTH} rows, got {len(ohlcv_window)}"
    )
    ohlcv = ohlcv_window[["open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)

    # --- Price channel ---
    # Candle area occupies full height (or top 50 rows if volume included)
    candle_rows = _IMG_HEIGHT - 14 if include_volume else _IMG_HEIGHT
    vol_rows = 14 if include_volume else 0

    price_canvas = np.zeros((_IMG_HEIGHT, _IMG_WIDTH), dtype=np.uint8)

    price_min = ohlcv[["open", "high", "low", "close"]].values.min()
    price_max = ohlcv[["open", "high", "low", "close"]].values.max()
    price_range = price_max - price_min
    if price_range == 0:
        price_range = 1.0  # flat price window -- avoid division by zero

    def price_to_row(p: float) -> int:
        """Map price to pixel row (0 = top = high price)."""
        norm = (p - price_min) / price_range
        row = int((1.0 - norm) * (candle_rows - 1))
        return max(0, min(candle_rows - 1, row))

    for day_idx, row in ohlcv.iterrows():
        col = int(day_idx)

        high_row = price_to_row(row["high"])
        low_row = price_to_row(row["low"])
        open_row = price_to_row(row["open"])
        close_row = price_to_row(row["close"])

        body_top = min(open_row, close_row)
        body_bot = max(open_row, close_row)

        # Wick: single-pixel column
        price_canvas[high_row:low_row + 1, col] = 255

        # Body: 3-pixel wide centered on column
        body_start_col = max(0, col - _CANDLE_BODY_WIDTH // 2)
        body_end_col = min(_IMG_WIDTH, col + _CANDLE_BODY_WIDTH // 2 + 1)
        price_canvas[body_top:body_bot + 1, body_start_col:body_end_col] = 255

    if not include_volume:
        return price_canvas[np.newaxis, :, :]  # (1, H, W)

    # --- Volume channel (bottom 14 rows of the full canvas) ---
    vol_canvas = np.zeros((_IMG_HEIGHT, _IMG_WIDTH), dtype=np.uint8)
    vol_vals = ohlcv["volume"].values.astype(float)
    vol_max = vol_vals.max()
    if vol_max == 0:
        vol_max = 1.0

    for day_idx, vol in enumerate(vol_vals):
        bar_height = int((vol / vol_max) * (vol_rows - 1))
        bar_top = _IMG_HEIGHT - bar_height
        vol_canvas[bar_top:, day_idx] = 255

    return np.stack([price_canvas, vol_canvas], axis=0)  # (2, H, W)


def _cache_path(ticker: str, date: pd.Timestamp) -> Path:
    return _CACHE_DIR / ticker / f"{date.strftime('%Y%m%d')}.npy"


def render_and_cache(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    window: int = 60,
    include_volume: bool = True,
    force_refresh: bool = False,
) -> dict[pd.Timestamp, Path]:
    """Render and cache all valid decision-date images for one ticker.

    For each date t that has at least `window` prior trading days, renders
    the trailing `window`-day OHLCV window and saves to cache.

    Parameters
    ----------
    ticker:
        Ticker symbol (used for cache subdirectory).
    ohlcv_df:
        Full OHLCV DataFrame for the ticker, sorted ascending by date.
    window:
        Number of trailing trading days per image (default 60).
    include_volume:
        Include volume as a second channel.
    force_refresh:
        Re-render even if cache file exists.

    Returns
    -------
    dict[pd.Timestamp, Path]
        Mapping from decision date to the path of the saved .npy file.
    """
    ticker_cache = _CACHE_DIR / ticker
    ticker_cache.mkdir(parents=True, exist_ok=True)

    ohlcv_df = ohlcv_df.sort_index()
    dates = ohlcv_df.index

    saved: dict[pd.Timestamp, Path] = {}
    n_rendered = 0

    for i in range(window - 1, len(dates)):
        date = dates[i]
        path = _cache_path(ticker, date)

        if not force_refresh and path.exists():
            saved[date] = path
            continue

        window_df = ohlcv_df.iloc[i - window + 1 : i + 1]
        if len(window_df) < window:
            continue

        img = _render_single_image(window_df, include_volume=include_volume)
        np.save(path, img)
        saved[date] = path
        n_rendered += 1

    logger.info(
        "%s: %d images rendered, %d loaded from cache (total %d)",
        ticker, n_rendered, len(saved) - n_rendered, len(saved),
    )
    return saved


def load_image(ticker: str, date: pd.Timestamp) -> Optional[np.ndarray]:
    """Load a cached candlestick image array for (ticker, date).

    Returns
    -------
    np.ndarray or None
        Shape (C, H, W), or None if not cached.
    """
    path = _cache_path(ticker, date)
    if not path.exists():
        logger.warning("No cached image for %s on %s", ticker, date)
        return None
    return np.load(path)


def produce_sample_images(
    tickers: list[str],
    ohlcv_dict: dict[str, pd.DataFrame],
    n_samples: int = 5,
    include_volume: bool = True,
    seed: int = 42,
) -> list[Path]:
    """Render and save N random (ticker, date) sample images for visual QA.

    Saves PNG files to results/figures/candle_samples/ for manual inspection.
    Call this before training the CNN to confirm rendering is correct.

    Returns
    -------
    list[Path]
        Paths of the saved PNG files.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for sample image rendering") from e

    _SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Build candidate (ticker, date_index) pairs
    candidates: list[tuple[str, int]] = []
    for ticker in tickers:
        df = ohlcv_dict.get(ticker)
        if df is not None and len(df) >= 60:
            candidates.extend([(ticker, i) for i in range(60, len(df))])

    samples = rng.sample(candidates, min(n_samples, len(candidates)))
    saved_paths: list[Path] = []

    for ticker, idx in samples:
        df = ohlcv_dict[ticker]
        date = df.index[idx]
        window_df = df.iloc[idx - 60 + 1 : idx + 1]
        img = _render_single_image(window_df, include_volume=include_volume)

        fig, axes = plt.subplots(1, img.shape[0], figsize=(10, 4))
        if img.shape[0] == 1:
            axes = [axes]
        titles = ["OHLC Candles", "Volume"] if img.shape[0] == 2 else ["OHLC Candles"]
        for ax, channel, title in zip(axes, img, titles):
            ax.imshow(channel, cmap="gray", origin="upper", aspect="auto")
            ax.set_title(f"{ticker} {date.date()} — {title}")
            ax.axis("off")

        fname = f"candle_{ticker}_{date.strftime('%Y%m%d')}.png"
        out_path = _SAMPLE_DIR / fname
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
        logger.info("Saved sample image: %s", out_path)

    return saved_paths
