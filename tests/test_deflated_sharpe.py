"""Tests for src/evaluation/deflated_sharpe.py."""
import numpy as np
import pytest

from src.evaluation.deflated_sharpe import (
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    _sharpe_se,
)


class TestProbabilisticSharpeRatio:
    def test_observed_equals_benchmark_returns_half(self):
        """PSR(SR* = SR̂) = 0.5 by symmetry of the normal CDF."""
        psr = probabilistic_sharpe_ratio(
            observed_sr=1.0, benchmark_sr=1.0, n_obs=252
        )
        assert abs(psr - 0.5) < 1e-9

    def test_above_benchmark_above_half(self):
        psr = probabilistic_sharpe_ratio(
            observed_sr=2.0, benchmark_sr=1.0, n_obs=252
        )
        assert psr > 0.5

    def test_below_benchmark_below_half(self):
        psr = probabilistic_sharpe_ratio(
            observed_sr=0.5, benchmark_sr=1.0, n_obs=252
        )
        assert psr < 0.5

    def test_larger_sample_gives_tighter_psr(self):
        """Same SR gap: more data → higher confidence SR > SR*."""
        psr_small = probabilistic_sharpe_ratio(1.5, 1.0, n_obs=100)
        psr_large = probabilistic_sharpe_ratio(1.5, 1.0, n_obs=1000)
        assert psr_large > psr_small

    def test_negative_skew_reduces_psr(self):
        """Negative skew widens SE → smaller z-score → lower PSR (use n=20 to avoid saturation)."""
        psr_normal = probabilistic_sharpe_ratio(0.5, 0.0, n_obs=20, skew=0.0)
        psr_negskew = probabilistic_sharpe_ratio(0.5, 0.0, n_obs=20, skew=-1.0)
        assert psr_negskew < psr_normal

    def test_small_n_obs_returns_nan(self):
        psr = probabilistic_sharpe_ratio(1.0, 0.0, n_obs=1)
        assert np.isnan(psr)


class TestDeflatedSharpeRatio:
    def test_n_trials_1_equals_psr_zero_benchmark(self):
        """With N=1 (no specification search), DSR = PSR(SR*=0)."""
        sr, n, skew, kurt = 1.5, 252, 0.0, 0.0
        from src.evaluation.deflated_sharpe import _sharpe_se
        se = _sharpe_se(sr, n, skew, kurt)
        sr_var = se ** 2
        dsr = deflated_sharpe_ratio(sr, n, n_trials=1, sr_variance=sr_var, skew=skew, kurt=kurt)
        psr = probabilistic_sharpe_ratio(sr, 0.0, n, skew=skew, kurt=kurt)
        assert abs(dsr - psr) < 1e-9

    def test_more_trials_deflates_more(self):
        """More specification search → higher expected max → lower DSR."""
        # Use SR=0.3, n=50 so DSR is in (0,1) and not near saturation
        sr, n = 0.3, 50
        se = _sharpe_se(sr, n)
        sr_var = se ** 2
        dsr_few = deflated_sharpe_ratio(sr, n, n_trials=2, sr_variance=sr_var)
        dsr_many = deflated_sharpe_ratio(sr, n, n_trials=16, sr_variance=sr_var)
        assert dsr_few > dsr_many

    def test_returns_probability_in_unit_interval(self):
        se = _sharpe_se(1.0, 500)
        dsr = deflated_sharpe_ratio(1.0, 500, n_trials=8, sr_variance=se ** 2)
        assert 0.0 <= dsr <= 1.0

    def test_nan_sr_variance_returns_nan(self):
        dsr = deflated_sharpe_ratio(1.0, 252, n_trials=8, sr_variance=np.nan)
        assert np.isnan(dsr)

    def test_n_trials_8_deflates_vs_n_trials_1(self):
        """8 configs → expected max > 0 → harder benchmark → DSR(N=8) < DSR(N=1)."""
        sr, n = 0.3, 50
        se = _sharpe_se(sr, n)
        sr_var = se ** 2
        dsr1 = deflated_sharpe_ratio(sr, n, n_trials=1, sr_variance=sr_var)
        dsr8 = deflated_sharpe_ratio(sr, n, n_trials=8, sr_variance=sr_var)
        assert dsr8 < dsr1


class TestMinTrackRecordLength:
    def test_increases_with_target_probability(self):
        """Achieving 99 % confidence requires more data than 95 %."""
        trl_95 = min_track_record_length(1.5, 0.0, prob_target=0.95)
        trl_99 = min_track_record_length(1.5, 0.0, prob_target=0.99)
        assert trl_99 > trl_95

    def test_returns_inf_when_observed_leq_benchmark(self):
        trl = min_track_record_length(0.5, 1.0)
        assert np.isinf(trl)

    def test_larger_sr_gap_reduces_trl(self):
        trl_small_gap = min_track_record_length(1.1, 1.0)
        trl_large_gap = min_track_record_length(2.0, 1.0)
        assert trl_large_gap < trl_small_gap

    def test_positive_result_for_typical_case(self):
        trl = min_track_record_length(2.0, 0.0, prob_target=0.95)
        assert trl > 0
        assert np.isfinite(trl)
