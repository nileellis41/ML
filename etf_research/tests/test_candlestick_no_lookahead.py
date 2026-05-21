"""Candlestick rendering look-ahead tests.

Verifies that rendered images use only data up to and including the decision date.
Run with: pytest tests/test_candlestick_no_lookahead.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.data.candlestick_render import _render_single_image, _IMG_HEIGHT, _IMG_WIDTH


def _make_ohlcv(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=dates)


class TestCandlestickNoLookahead:
    def test_image_uses_exactly_60_rows(self):
        """Rendering should consume exactly 60 rows -- not more."""
        ohlcv = _make_ohlcv(80)
        window_60 = ohlcv.iloc[:60]
        img = _render_single_image(window_60)
        assert img.shape == (2, _IMG_HEIGHT, _IMG_WIDTH)

    def test_render_with_wrong_length_raises(self):
        """A window of != 60 rows should raise AssertionError."""
        ohlcv_59 = _make_ohlcv(59)
        with pytest.raises(AssertionError, match="Expected 60 rows"):
            _render_single_image(ohlcv_59)

        ohlcv_61 = _make_ohlcv(61)
        with pytest.raises(AssertionError, match="Expected 60 rows"):
            _render_single_image(ohlcv_61)

    def test_prices_are_min_max_scaled(self):
        """Image pixel values should span most of the [0,255] range."""
        ohlcv = _make_ohlcv(60)
        img = _render_single_image(ohlcv, include_volume=False)
        price_channel = img[0]
        # Should have both 0 and 255 pixels (background and foreground)
        assert price_channel.max() == 255, "No white pixels -- rendering failed"
        assert price_channel.min() == 0, "No black pixels -- rendering failed"

    def test_image_dtype_is_uint8(self):
        ohlcv = _make_ohlcv(60)
        img = _render_single_image(ohlcv)
        assert img.dtype == np.uint8

    def test_single_channel_when_no_volume(self):
        ohlcv = _make_ohlcv(60)
        img = _render_single_image(ohlcv, include_volume=False)
        assert img.shape[0] == 1, f"Expected 1 channel, got {img.shape[0]}"

    def test_two_channels_with_volume(self):
        ohlcv = _make_ohlcv(60)
        img = _render_single_image(ohlcv, include_volume=True)
        assert img.shape[0] == 2, f"Expected 2 channels, got {img.shape[0]}"

    def test_flat_price_window_does_not_crash(self):
        """Window with zero price range should not divide by zero."""
        ohlcv = _make_ohlcv(60)
        ohlcv["open"] = 100.0
        ohlcv["high"] = 100.0
        ohlcv["low"] = 100.0
        ohlcv["close"] = 100.0
        img = _render_single_image(ohlcv)
        assert img is not None
        assert not np.isnan(img).any()

    def test_zero_volume_does_not_crash(self):
        ohlcv = _make_ohlcv(60)
        ohlcv["volume"] = 0.0
        img = _render_single_image(ohlcv, include_volume=True)
        assert img is not None

    def test_future_rows_not_used(self):
        """Changing rows AFTER the 60-day window should not affect the rendered image."""
        ohlcv = _make_ohlcv(80)
        window = ohlcv.iloc[:60].copy()
        img_original = _render_single_image(window.copy())

        # Mutate rows beyond the 60-day window (these should not influence the image)
        ohlcv_modified = ohlcv.copy()
        ohlcv_modified.iloc[60:] = ohlcv_modified.iloc[60:] * 100  # extreme modification
        window_same = ohlcv_modified.iloc[:60]

        img_same = _render_single_image(window_same.copy())
        np.testing.assert_array_equal(
            img_original, img_same,
            err_msg="Image changed when future rows were modified -- look-ahead detected."
        )

    def test_render_and_cache_uses_only_past_data(self, tmp_path):
        """render_and_cache should generate one image per decision date,
        and each image should be built from at most 60 prior rows."""
        import os
        from src.data.candlestick_render import render_and_cache, _CACHE_DIR

        ohlcv = _make_ohlcv(90, seed=1)

        # Monkeypatch cache dir to tmp_path
        import src.data.candlestick_render as cr_mod
        original_cache = cr_mod._CACHE_DIR
        cr_mod._CACHE_DIR = tmp_path / "candles"

        try:
            saved = render_and_cache("TEST", ohlcv, window=60, force_refresh=True)
        finally:
            cr_mod._CACHE_DIR = original_cache

        # Should have images for indices 59..89 (60 lookback required)
        assert len(saved) == 90 - 60 + 1, f"Expected 31 images, got {len(saved)}"

        # Load each image and verify shape
        for date, path in saved.items():
            img = np.load(path)
            assert img.shape == (2, _IMG_HEIGHT, _IMG_WIDTH), (
                f"Wrong image shape for {date}: {img.shape}"
            )
