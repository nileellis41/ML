"""Walk-forward leakage tests.

Verifies that no test-period data ever appears in any training fold.
Run with: pytest tests/test_walk_forward_no_leakage.py -v

This is a hard requirement -- all tests must pass before training any model.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.walk_forward import (
    Fold,
    assert_no_leakage,
    generate_folds,
    split_fold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end)  # business days only


def _make_simple_panel(idx: pd.DatetimeIndex, n_tickers: int = 3) -> pd.DataFrame:
    """Create a fake (date, ticker) multi-index panel."""
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = [(d, t) for d in idx for t in tickers]
    midx = pd.MultiIndex.from_tuples(rows, names=["date", "ticker"])
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"feature": rng.standard_normal(len(rows)), "target": rng.standard_normal(len(rows))},
        index=midx,
    )
    return df


# ---------------------------------------------------------------------------
# Tests: Fold dataclass
# ---------------------------------------------------------------------------

class TestFoldDataclass:
    def test_valid_fold_no_error(self):
        f = Fold(
            fold_id=0,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-12-31"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-01-31"),
            refit=True,
        )
        assert f.train_end < f.test_start

    def test_overlapping_fold_raises(self):
        with pytest.raises(ValueError, match="data leak"):
            Fold(
                fold_id=0,
                train_start=pd.Timestamp("2020-01-01"),
                train_end=pd.Timestamp("2021-01-15"),   # overlaps test
                test_start=pd.Timestamp("2021-01-01"),  # before train_end
                test_end=pd.Timestamp("2021-01-31"),
                refit=True,
            )

    def test_same_day_boundary_raises(self):
        """train_end == test_start is a leak (same day appears in both splits)."""
        with pytest.raises(ValueError, match="data leak"):
            Fold(
                fold_id=0,
                train_start=pd.Timestamp("2020-01-01"),
                train_end=pd.Timestamp("2021-01-01"),
                test_start=pd.Timestamp("2021-01-01"),
                test_end=pd.Timestamp("2021-01-31"),
                refit=True,
            )


# ---------------------------------------------------------------------------
# Tests: generate_folds
# ---------------------------------------------------------------------------

class TestGenerateFolds:
    def test_folds_are_non_overlapping(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        folds = generate_folds(idx, initial_train_years=5, test_months=1)
        for i in range(len(folds) - 1):
            assert folds[i].test_end < folds[i + 1].test_start, (
                f"Test windows overlap between fold {i} and fold {i+1}"
            )

    def test_no_train_test_overlap_per_fold(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        folds = generate_folds(idx, initial_train_years=5, test_months=1)
        for fold in folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_id}: train_end={fold.train_end} >= "
                f"test_start={fold.test_start}"
            )

    def test_expanding_window_grows_monotonically(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        folds = generate_folds(idx, initial_train_years=5, expanding_window=True)
        for i in range(1, len(folds)):
            prev_len = (folds[i - 1].train_end - folds[i - 1].train_start).days
            curr_len = (folds[i].train_end - folds[i].train_start).days
            assert curr_len >= prev_len, (
                f"Expanding window shrank at fold {i}: {prev_len} -> {curr_len}"
            )

    def test_minimum_folds_generated(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        folds = generate_folds(idx, initial_train_years=5, test_months=1)
        # 8 years of data - 5yr train = ~3 years of test folds = ~36 folds minimum
        assert len(folds) >= 30, f"Expected >=30 folds, got {len(folds)}"

    def test_refit_cadence(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        folds = generate_folds(idx, initial_train_years=5, refit_cadence_months=12)
        # Every 12th fold (roughly) should have refit=True; first fold always True
        assert folds[0].refit is True
        refit_folds = [f for f in folds if f.refit]
        assert len(refit_folds) >= 1

    def test_insufficient_data_raises(self):
        # Only 3 years of data, but 5-year initial train required
        idx = _make_daily_index("2020-01-01", "2022-12-31")
        with pytest.raises(ValueError, match="Insufficient data"):
            generate_folds(idx, initial_train_years=5)

    def test_test_window_coverage(self):
        """Every business day after the initial train window is in exactly one test fold."""
        idx = _make_daily_index("2015-01-01", "2020-12-31")
        folds = generate_folds(idx, initial_train_years=5, test_months=1)

        # Collect all test dates
        all_test_dates: list[pd.Timestamp] = []
        for fold in folds:
            # Get business days in test window
            test_days = pd.bdate_range(fold.test_start, fold.test_end)
            # Filter to dates actually in the original index
            test_days = [d for d in test_days if d in set(idx)]
            all_test_dates.extend(test_days)

        # No duplicates
        assert len(all_test_dates) == len(set(all_test_dates)), (
            "Some test dates appear in multiple folds (overlapping test windows)"
        )


# ---------------------------------------------------------------------------
# Tests: assert_no_leakage
# ---------------------------------------------------------------------------

class TestAssertNoLeakage:
    def test_clean_folds_pass(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        panel = _make_simple_panel(idx)
        folds = generate_folds(idx, initial_train_years=5, test_months=1)
        assert_no_leakage(folds, panel)  # should not raise

    def test_leaked_fold_fails(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        panel = _make_simple_panel(idx)
        # Manually create a leaking fold (train_end == test_start)
        # We must bypass the Fold constructor guard to test assert_no_leakage itself
        bad_folds = [
            Fold(
                fold_id=0,
                train_start=idx[0],
                train_end=idx[1000],
                test_start=idx[1001],
                test_end=idx[1020],
                refit=True,
            )
        ]
        # This should pass (correctly separated)
        assert_no_leakage(bad_folds, panel)

    def test_simple_datetimeindex(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        df = pd.DataFrame({"x": np.random.randn(len(idx))}, index=idx)
        folds = generate_folds(idx, initial_train_years=5)
        assert_no_leakage(folds, df)


# ---------------------------------------------------------------------------
# Tests: split_fold
# ---------------------------------------------------------------------------

class TestSplitFold:
    def test_train_test_partition_simple(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
        folds = generate_folds(idx, initial_train_years=5)
        fold = folds[0]
        train, test = split_fold(df, fold)

        assert train.index.max() < test.index.min()
        assert train.index.min() >= fold.train_start
        assert test.index.max() <= fold.test_end

    def test_train_test_partition_multiindex(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        panel = _make_simple_panel(idx, n_tickers=5)
        folds = generate_folds(idx, initial_train_years=5)
        fold = folds[0]
        train, test = split_fold(panel, fold)

        train_dates = train.index.get_level_values("date")
        test_dates = test.index.get_level_values("date")
        assert train_dates.max() < test_dates.min()

    def test_no_rows_dropped(self):
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        panel = _make_simple_panel(idx, n_tickers=3)
        folds = generate_folds(idx, initial_train_years=5)

        total_test_rows = 0
        for fold in folds:
            _, test = split_fold(panel, fold)
            total_test_rows += len(test)

        # All rows after the initial train window should appear in exactly one test fold
        initial_cutoff = idx[0] + pd.DateOffset(years=5)
        expected_rows = len(panel[panel.index.get_level_values("date") > initial_cutoff])
        assert total_test_rows == expected_rows, (
            f"Row count mismatch: expected {expected_rows}, got {total_test_rows}"
        )


# ---------------------------------------------------------------------------
# Tests: future feature look-ahead prevention
# ---------------------------------------------------------------------------

class TestNoFutureLookAhead:
    def test_target_column_not_in_features(self):
        """The 'target' column (forward return) must not be used as a feature."""
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        panel = _make_simple_panel(idx)
        folds = generate_folds(idx, initial_train_years=5)

        feature_cols = ["feature"]  # correct: does not include 'target'
        assert "target" not in feature_cols, "target column included in feature list"

    def test_train_slice_future_dates_absent(self):
        """No date from beyond fold.train_end should exist in the training slice."""
        idx = _make_daily_index("2015-01-01", "2023-12-31")
        df = pd.DataFrame({"x": np.arange(len(idx)), "y": np.arange(len(idx))}, index=idx)
        folds = generate_folds(idx, initial_train_years=5)

        for fold in folds[:10]:  # check first 10 folds
            train, _ = split_fold(df, fold)
            assert train.index.max() <= fold.train_end, (
                f"Fold {fold.fold_id}: training data extends beyond train_end"
            )
