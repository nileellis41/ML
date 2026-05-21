"""FRED series validity tests.

Tests that validate_fred.py correctly identifies valid and invalid series.
Run with: pytest tests/test_fred_series_validity.py -v

NOTE: These tests make network calls to FRED API.
Set FRED_API_KEY before running. Skip with: pytest -m "not fred"
"""
import os

import pytest
import yaml

# Mark all tests in this file as requiring FRED API
pytestmark = pytest.mark.fred


def _has_fred_key():
    return bool(os.environ.get("FRED_API_KEY"))


@pytest.fixture
def features_config():
    with open("config/features.yaml") as f:
        return yaml.safe_load(f)


class TestFredSeriesIds:
    @pytest.mark.skipif(not _has_fred_key(), reason="FRED_API_KEY not set")
    def test_validate_all_series_do_not_raise(self):
        """Running validate_all() on the config should complete without SystemExit."""
        from src.data.validate_fred import validate_all
        try:
            results = validate_all(force_refresh=False)
        except SystemExit as e:
            pytest.fail(f"validate_all() exited with code {e.code}. Some FRED series are invalid.")

        failed = [sid for sid, ok in results.items() if not ok]
        assert not failed, f"The following FRED series failed: {failed}"

    def test_all_series_ids_are_strings(self, features_config):
        """Every series ID in features.yaml should be a non-empty string."""
        from src.data.validate_fred import _collect_all_series_ids
        ids = _collect_all_series_ids(features_config)
        for sid in ids:
            assert isinstance(sid, str) and len(sid) > 0, f"Bad series ID: {sid!r}"

    def test_no_duplicate_series_in_universal(self, features_config):
        """Universal macro block should not contain duplicate series IDs."""
        ids = features_config.get("universal_macro", {}).get("fred", [])
        assert len(ids) == len(set(ids)), f"Duplicate IDs in universal_macro: {ids}"

    def test_sector_features_each_have_fred_list(self, features_config):
        """Every entry in sector_features should have a 'fred' key with a list."""
        sector_features = features_config.get("sector_features", {})
        for etf, block in sector_features.items():
            assert "fred" in block or "cross_asset" in block, (
                f"{etf} has neither 'fred' nor 'cross_asset' in its sector block."
            )
            if "fred" in block:
                assert isinstance(block["fred"], list), f"{etf}.fred is not a list"

    @pytest.mark.skipif(not _has_fred_key(), reason="FRED_API_KEY not set")
    def test_known_valid_series(self):
        """A few canonical series known to exist should load without error."""
        from src.data.fred_loader import load_series
        known_valid = ["DGS10", "VIXCLS", "UNRATE", "DFF"]
        for sid in known_valid:
            s = load_series(sid, start="2020-01-01", force_refresh=False)
            assert len(s) > 0, f"{sid} returned empty series"
            assert s.dropna().sum() != 0, f"{sid} is all NaN"

    @pytest.mark.skipif(not _has_fred_key(), reason="FRED_API_KEY not set")
    def test_invalid_series_raises(self):
        """A made-up series ID should raise RuntimeError (not silently return empty)."""
        from src.data.fred_loader import load_series
        with pytest.raises(RuntimeError, match="FRED"):
            load_series("XXXXNOTASERIESID", start="2020-01-01", force_refresh=True)
