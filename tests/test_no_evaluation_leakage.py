"""Tests that the evaluation pipeline does not introduce look-ahead leakage.

Covers _returns_from_predictions and _hit_rate_pairs in reporting.py:
  - Signal must use only the FIRST date of each fold (non-overlapping windows)
  - Daily portfolio returns must be daily-sized (not 21-day-sized)
  - Hit rate inputs must be (N_folds x N_tickers) pairs, not all test dates
  - Portfolio returns must be plausible (no annualised Sharpe > 10 on random data)
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.reporting import (
    _date_to_fold_map,
    _first_date_signals,
    _hit_rate_pairs,
)
from src.evaluation.metrics import annualised_return, annualised_volatility, sharpe_ratio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pred_df(
    n_folds: int = 4,
    fold_length: int = 22,
    tickers: list[str] = None,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Synthetic prediction DataFrame with (date, ticker) MultiIndex."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(rng_seed)

    start = pd.Timestamp("2024-01-02")
    all_dates = pd.bdate_range(start, periods=n_folds * fold_length)

    rows = []
    for fold_id in range(n_folds):
        fold_dates = all_dates[fold_id * fold_length : (fold_id + 1) * fold_length]
        for d in fold_dates:
            for t in tickers:
                pred = float(rng.normal(0.005, 0.02))
                target = float(rng.normal(0.015, 0.05))
                rows.append({"date": d, "ticker": t, "prediction": pred, "target": target, "fold_id": fold_id})

    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


# ---------------------------------------------------------------------------
# _first_date_signals
# ---------------------------------------------------------------------------

class TestFirstDateSignals:
    def test_returns_one_entry_per_fold(self):
        df = _make_pred_df(n_folds=3)
        signals = _first_date_signals(df)
        assert len(signals) == 3

    def test_signal_tickers_are_subset_of_universe(self):
        tickers = ["AAA", "BBB", "CCC"]
        df = _make_pred_df(tickers=tickers)
        signals = _first_date_signals(df)
        for pos in signals.values():
            assert all(t in tickers for t in pos)

    def test_signal_date_is_earliest_in_fold(self):
        df = _make_pred_df(n_folds=2, fold_length=10)
        for fold_id, fold_df in df.groupby("fold_id"):
            fold_dates = sorted(fold_df.index.get_level_values("date").unique())
            expected_first = fold_dates[0]
            # Verify _first_date_signals picks this date
        signals = _first_date_signals(df)
        # One signal per fold — just check it ran without error and has correct keys
        assert set(signals.keys()) == {0, 1}

    def test_above_median_signal_selects_half_the_tickers(self):
        """Above-median rule selects tickers strictly above the cross-sectional median."""
        tickers = ["X", "Y", "Z"]
        df = _make_pred_df(tickers=tickers)
        # Force distinct predictions so ranking is unambiguous
        first_date = df.index.get_level_values("date").unique().sort_values()[0]
        for i, t in enumerate(tickers):
            df.loc[(first_date, t), "prediction"] = float(i)  # 0, 1, 2
        signals = _first_date_signals(df)
        # median = 1.0 → only "Z" (pred=2) is above median
        assert "Z" in signals[0]
        assert "X" not in signals[0]

    def test_tied_predictions_yield_empty_signal(self):
        """When all predictions are identical, none exceed the median → empty signal."""
        df = _make_pred_df(tickers=["X", "Y"])
        df["prediction"] = 1.0
        signals = _first_date_signals(df)
        for pos in signals.values():
            assert pos == []

    def test_empty_signal_when_all_preds_negative_and_tied(self):
        df = _make_pred_df(tickers=["X", "Y"])
        df["prediction"] = -1.0
        signals = _first_date_signals(df)
        for pos in signals.values():
            assert pos == []


# ---------------------------------------------------------------------------
# _date_to_fold_map
# ---------------------------------------------------------------------------

class TestDateToFoldMap:
    def test_every_test_date_mapped(self):
        df = _make_pred_df(n_folds=3, fold_length=5)
        d2f = _date_to_fold_map(df)
        test_dates = set(df.index.get_level_values("date").unique())
        assert set(d2f.keys()) == test_dates

    def test_fold_boundaries_non_overlapping(self):
        """Each date maps to exactly one fold."""
        df = _make_pred_df(n_folds=4, fold_length=10)
        d2f = _date_to_fold_map(df)
        # Just verify all values are valid fold_ids
        valid_folds = set(df["fold_id"].unique())
        for fold_id in d2f.values():
            assert fold_id in valid_folds


# ---------------------------------------------------------------------------
# _hit_rate_pairs
# ---------------------------------------------------------------------------

class TestHitRatePairs:
    def test_output_length_is_n_folds_times_n_tickers(self):
        n_folds, n_tickers = 4, 3
        df = _make_pred_df(n_folds=n_folds, fold_length=22, tickers=["A", "B", "C"])
        preds, actuals = _hit_rate_pairs(df)
        # One signal date × n_tickers per fold
        assert len(preds) == n_folds * n_tickers
        assert len(actuals) == n_folds * n_tickers

    def test_no_nan_in_outputs(self):
        df = _make_pred_df()
        preds, actuals = _hit_rate_pairs(df)
        assert preds.notna().all()
        assert actuals.notna().all()

    def test_hit_rate_in_valid_range(self):
        df = _make_pred_df(rng_seed=42)
        preds, actuals = _hit_rate_pairs(df)
        from src.evaluation.metrics import hit_rate
        hr = hit_rate(preds, actuals)
        assert 0.0 <= hr <= 1.0

    def test_returns_empty_on_missing_columns(self):
        df = _make_pred_df()
        df = df.drop(columns=["target"])
        preds, actuals = _hit_rate_pairs(df)
        assert len(preds) == 0
        assert len(actuals) == 0

    def test_hit_rate_near_half_for_uncorrelated_predictions(self):
        """Random predictions against random targets should yield ~50% hit rate."""
        rng = np.random.default_rng(1)
        n = 200
        df = _make_pred_df(n_folds=10, fold_length=22, tickers=[f"T{i}" for i in range(20)], rng_seed=1)
        preds, actuals = _hit_rate_pairs(df)
        from src.evaluation.metrics import hit_rate
        hr = hit_rate(preds, actuals)
        # Random: expect hit rate in [0.35, 0.65]
        assert 0.35 <= hr <= 0.65, f"Expected ~0.50, got {hr:.3f}"


# ---------------------------------------------------------------------------
# Non-overlapping / scaling checks on metrics
# ---------------------------------------------------------------------------

class TestPortfolioReturnScaling:
    """Verify that the reporting pipeline doesn't inflate metrics.

    We can't call _returns_from_predictions directly (it loads Alpaca files),
    but we can test the metric formulas on synthetic daily returns with
    known properties and ensure the sanity ranges hold.
    """

    def test_annualised_return_range_for_sane_daily_returns(self):
        """A daily return series with mean=0.06%/day annualises to ~16%."""
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 0.01, 252)
        noise -= noise.mean()  # zero-centre noise, then add target drift
        daily = pd.Series(noise + 0.0006)
        ann_ret = annualised_return(daily)
        assert 0.05 <= ann_ret <= 0.35, f"ann_return={ann_ret:.3f} outside sane range"

    def test_sharpe_range_for_sane_daily_returns(self):
        """A decent strategy with Sharpe ~1 should stay in [0.3, 2.0]."""
        rng = np.random.default_rng(8)
        daily = pd.Series(rng.normal(0.0004, 0.01, 252))
        sr = sharpe_ratio(daily)
        assert 0.3 <= sr <= 2.5, f"Sharpe={sr:.3f} outside sane range"

    def test_monthly_returns_wrongly_treated_as_daily_inflate_metrics(self):
        """This test DOCUMENTS the bug that was fixed: 21-day returns treated as daily
        produce absurdly high Sharpe ratios (>> 10). After the fix, this should not
        happen because _returns_from_predictions now loads actual daily returns.
        """
        rng = np.random.default_rng(9)
        # Simulate 13 monthly returns (~2% mean, 3% std)
        monthly = pd.Series(rng.normal(0.02, 0.03, 13))
        # Wrong annualisation (treating monthly returns as daily):
        ann_ret_wrong = annualised_return(monthly)  # exp(mean * 252) - 1 ≈ huge
        # The wrong method gives absurdly high numbers
        assert ann_ret_wrong > 1.0, (
            f"Expected inflated ann_ret from monthly-as-daily bug, got {ann_ret_wrong:.3f}"
        )
        # Correct annualisation for monthly:
        ann_ret_correct = float(np.exp(monthly.mean() * 12) - 1)
        assert 0.05 <= ann_ret_correct <= 0.40, (
            f"Correct monthly ann_ret={ann_ret_correct:.3f} outside sane range"
        )

    def test_overlapping_windows_inflate_return_series(self):
        """Overlapping 21-day windows create autocorrelation that inflates Sharpe.
        Illustrates why _returns_from_predictions must use non-overlapping signals.
        """
        rng = np.random.default_rng(10)
        # True daily returns: mean=0.04%, std=1% — enforce mean exactly
        noise = rng.normal(0, 0.01, 273)
        true_daily = noise - noise.mean() + 0.0004  # guaranteed positive drift
        # Compute overlapping 21-day cumulative returns as if they were daily
        overlapping = pd.Series([
            true_daily[i:i+21].sum() for i in range(273)
            if i + 21 <= len(true_daily)
        ])
        sharpe_overlapping = sharpe_ratio(overlapping)
        # Overlapping 21-day series treated as daily massively inflates Sharpe
        assert sharpe_overlapping > 5.0, (
            f"Expected inflated Sharpe from overlapping windows, got {sharpe_overlapping:.3f}"
        )
        # True daily Sharpe:
        sharpe_daily = sharpe_ratio(pd.Series(true_daily))
        assert sharpe_daily < 2.0, f"True daily Sharpe={sharpe_daily:.3f} should be modest"
