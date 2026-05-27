"""Tests for src/evaluation/significance.py."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.significance import diebold_mariano_test, run_significance_tests


def _make_returns(seed: int, n: int = 300, mu: float = 0.0, sig: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mu, sig, n), index=idx)


class TestDieboldMarianoTest:
    def test_identical_returns_zero_stat(self):
        """When returns are identical, DM stat should be 0."""
        r = _make_returns(0)
        _, _, mdm, pval = diebold_mariano_test(r, r)
        assert abs(mdm) < 1e-9
        assert abs(pval - 1.0) < 1e-6

    def test_clearly_different_returns_significant(self):
        """Large, consistent mean difference (n=2000) should be significant."""
        # Use shared index so returns differ only in mean, not sample noise
        rng = np.random.default_rng(99)
        idx = pd.date_range("2015-01-01", periods=2000, freq="B")
        noise = rng.normal(0, 0.01, 2000)
        r_good = pd.Series(noise + 0.003, index=idx)   # +75% ann
        r_bad = pd.Series(noise - 0.003, index=idx)    # -75% ann
        _, _, mdm, pval = diebold_mariano_test(r_good, r_bad)
        assert pval < 0.001

    def test_mean_diff_is_annualised(self):
        """Use same-noise base so sample means differ by exactly mu_a - mu_b."""
        mu_a, mu_b = 0.001, 0.0005
        rng = np.random.default_rng(42)
        idx = pd.date_range("2020-01-01", periods=500, freq="B")
        noise = rng.normal(0, 0.01, 500)
        r_a = pd.Series(noise + mu_a, index=idx)
        r_b = pd.Series(noise + mu_b, index=idx)
        mean_diff, _, _, _ = diebold_mariano_test(r_a, r_b)
        expected = (mu_a - mu_b) * 252
        assert abs(mean_diff - expected) < 1e-9  # exact to floating-point precision

    def test_too_few_observations_returns_nan(self):
        r_a = _make_returns(0, n=5)
        r_b = _make_returns(1, n=5)
        result = diebold_mariano_test(r_a, r_b)
        assert all(np.isnan(v) for v in result)

    def test_nonoverlapping_indices_returns_nan(self):
        r_a = pd.Series([0.01] * 50, index=pd.date_range("2022-01-01", periods=50, freq="B"))
        r_b = pd.Series([0.01] * 50, index=pd.date_range("2024-01-01", periods=50, freq="B"))
        result = diebold_mariano_test(r_a, r_b)
        assert all(np.isnan(v) for v in result)

    def test_sharpe_diff_sign(self):
        """Positive mean diff should give positive Sharpe diff."""
        r_a = _make_returns(0, mu=0.001, sig=0.01)
        r_b = _make_returns(0, mu=-0.001, sig=0.01)
        _, sharpe_diff, _, _ = diebold_mariano_test(r_a, r_b)
        assert sharpe_diff > 0


class TestRunSignificanceTests:
    def _build_returns(self) -> tuple[dict, dict]:
        ml = {}
        bl = {}
        names = ["linear_base", "linear_regime", "logistic_base", "logistic_regime",
                 "pca_base", "pca_regime", "lstm_base", "lstm_regime"]
        for i, name in enumerate(names):
            ml[name] = _make_returns(i, mu=0.0003)
        for j, bl_name in enumerate(["spy_bh", "equal_weight", "momentum_12_1", "sixty_forty"]):
            bl[bl_name] = _make_returns(10 + j, mu=0.0003)
        return ml, bl

    def test_returns_32_rows(self):
        ml, bl = self._build_returns()
        df = run_significance_tests(ml, bl)
        assert len(df) == 32

    def test_expected_columns(self):
        ml, bl = self._build_returns()
        df = run_significance_tests(ml, bl)
        required = {
            "ml_model", "ml_variant", "baseline",
            "mean_return_diff_annual", "sharpe_diff",
            "dm_stat", "pvalue_raw", "pvalue_bonferroni",
            "pvalue_bh", "significant_at_05_after_bonferroni",
        }
        assert required.issubset(set(df.columns))

    def test_bonferroni_always_geq_raw(self):
        ml, bl = self._build_returns()
        df = run_significance_tests(ml, bl)
        valid = df["pvalue_raw"].notna()
        assert (df.loc[valid, "pvalue_bonferroni"] >= df.loc[valid, "pvalue_raw"]).all()

    def test_bh_pvals_in_unit_interval(self):
        ml, bl = self._build_returns()
        df = run_significance_tests(ml, bl)
        valid = df["pvalue_bh"].notna()
        assert (df.loc[valid, "pvalue_bh"] >= 0).all()
        assert (df.loc[valid, "pvalue_bh"] <= 1).all()

    def test_bonferroni_at_most_1(self):
        ml, bl = self._build_returns()
        df = run_significance_tests(ml, bl)
        valid = df["pvalue_bonferroni"].notna()
        assert (df.loc[valid, "pvalue_bonferroni"] <= 1.0).all()

    def test_models_and_variants_present(self):
        ml, bl = self._build_returns()
        df = run_significance_tests(ml, bl)
        assert set(df["ml_model"].unique()) == {"linear", "logistic", "pca", "lstm"}
        assert set(df["ml_variant"].unique()) == {"base", "regime"}
