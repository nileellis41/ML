"""Portfolio and prediction evaluation metrics.

All metrics are computed from return series -- no look-ahead, no regime probs
used unless explicitly passed for conditional decomposition.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def annualised_return(returns: pd.Series) -> float:
    """Geometric annualised return from daily log returns."""
    total = returns.sum()
    n = len(returns)
    if n == 0:
        return np.nan
    return float(np.exp(total * _TRADING_DAYS / n) - 1)


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily log returns."""
    return float(returns.std() * np.sqrt(_TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Daily log return series.
    rf:
        Annual risk-free rate (default 0).
    """
    mu = annualised_return(returns) - rf
    sigma = annualised_volatility(returns)
    return float(mu / sigma) if sigma > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a log return series."""
    cum = returns.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Annualised return / |max drawdown|."""
    ann_ret = annualised_return(returns)
    mdd = abs(max_drawdown(returns))
    return float(ann_ret / mdd) if mdd > 0 else np.nan


def turnover(weights: pd.DataFrame) -> float:
    """Average one-way portfolio turnover across rebalancing dates.

    Parameters
    ----------
    weights:
        DataFrame (dates x assets) of portfolio weights.

    Returns
    -------
    float
        Mean absolute weight change per period (one-way).
    """
    delta = weights.diff().abs().sum(axis=1).iloc[1:]
    return float(delta.mean())


def hit_rate(predictions: pd.Series, actuals: pd.Series) -> float:
    """Fraction of periods where predicted and actual return share the same sign."""
    valid = predictions.notna() & actuals.notna()
    if valid.sum() == 0:
        return np.nan
    return float(((predictions[valid] > 0) == (actuals[valid] > 0)).mean())


def sharpe_se(n_periods: int, sharpe_ann: float) -> float:
    """Standard error of an annualised Sharpe ratio.

    Uses 1/sqrt(n) * sqrt(252) — the asymptotic approximation under IID
    returns with zero mean.  Multiply by 1.96 for a 95 % CI half-width.
    """
    if n_periods <= 0:
        return np.nan
    return float(np.sqrt(_TRADING_DAYS / n_periods))


def stationary_bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 10_000,
    ci: float = 0.95,
    block_size: Optional[int] = None,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stationary bootstrap Sharpe confidence interval (Politis-Romano 1994).

    Block size defaults to max(1, round(T^(1/3))), which approximates the
    Politis-White (2004) optimal block length for typical financial return series.

    Parameters
    ----------
    returns   : daily log return series
    n_boot    : number of bootstrap replications
    ci        : confidence level (e.g. 0.95 for 95 %)
    block_size: mean block length; None uses T^(1/3)
    seed      : RNG seed for reproducibility

    Returns
    -------
    (sharpe_point, ci_lo, ci_hi)
    """
    r = returns.dropna().values.astype(float)
    n = len(r)
    if n < 20:
        sr = float(r.mean() / r.std(ddof=1) * _TRADING_DAYS ** 0.5) if r.std(ddof=1) > 0 else np.nan
        return sr, np.nan, np.nan

    if block_size is None:
        block_size = max(1, round(n ** (1 / 3)))
    p = 1.0 / block_size

    rng = np.random.default_rng(seed)

    # Build (n_boot, n) bootstrap index matrix via stationary bootstrap transitions.
    # At each step, with prob p start a new random block; otherwise advance by 1 (mod n).
    pos = np.empty((n_boot, n), dtype=np.int32)
    pos[:, 0] = rng.integers(0, n, size=n_boot)
    for t in range(1, n):
        new_block = rng.random(n_boot) < p
        new_starts = rng.integers(0, n, size=n_boot)
        pos[:, t] = np.where(new_block, new_starts, (pos[:, t - 1] + 1) % n)

    boot_r = r[pos]  # (n_boot, n)
    boot_mu = boot_r.mean(axis=1)
    boot_sig = boot_r.std(axis=1, ddof=1)
    valid = boot_sig > 0
    boot_sr = np.full(n_boot, np.nan)
    boot_sr[valid] = boot_mu[valid] / boot_sig[valid] * np.sqrt(_TRADING_DAYS)

    tail = (1.0 - ci) / 2.0
    ci_lo = float(np.nanpercentile(boot_sr, tail * 100))
    ci_hi = float(np.nanpercentile(boot_sr, (1.0 - tail) * 100))
    sr_pt = float(r.mean() / r.std(ddof=1) * np.sqrt(_TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan
    return sr_pt, ci_lo, ci_hi


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Information ratio: excess return over benchmark / tracking error."""
    excess = portfolio_returns - benchmark_returns
    te = excess.std() * np.sqrt(_TRADING_DAYS)
    if te == 0:
        return np.nan
    return float((excess.mean() * _TRADING_DAYS) / te)


def regime_conditional_metrics(
    returns: pd.Series,
    regime_labels: pd.Series,
    rf: float = 0.0,
) -> dict[str, dict]:
    """Compute Sharpe, return, and vol within each regime.

    Parameters
    ----------
    returns:
        Daily portfolio log return series (DatetimeIndex).
    regime_labels:
        Series of regime labels (DatetimeIndex, same calendar).
    rf:
        Annual risk-free rate.

    Returns
    -------
    dict
        Outer key = regime label, inner dict = metric_name -> value.
    """
    common_idx = returns.index.intersection(regime_labels.index)
    r = returns.reindex(common_idx)
    labels = regime_labels.reindex(common_idx)

    results: dict[str, dict] = {}
    for label in labels.unique():
        if pd.isna(label):
            continue
        mask = labels == label
        r_slice = r[mask]
        n = int(mask.sum())
        sr = sharpe_ratio(r_slice, rf=rf)
        se = sharpe_se(n, sr) if not np.isnan(sr) else np.nan
        results[str(label)] = {
            "ann_return": annualised_return(r_slice),
            "ann_vol": annualised_volatility(r_slice),
            "sharpe": sr,
            "sharpe_se": se,
            "sharpe_ci_lo": sr - 1.96 * se if not np.isnan(se) else np.nan,
            "sharpe_ci_hi": sr + 1.96 * se if not np.isnan(se) else np.nan,
            "max_drawdown": max_drawdown(r_slice),
            "n_periods": n,
        }
    return results


def compute_all_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    weights: Optional[pd.DataFrame] = None,
    predictions: Optional[pd.Series] = None,
    actuals: Optional[pd.Series] = None,
    regime_labels: Optional[pd.Series] = None,
    rf: float = 0.0,
) -> dict:
    """Compute the full metrics suite for one model-variant combination.

    Returns
    -------
    dict
        Flat dict of all metrics for insertion into the master comparison table.
    """
    m: dict = {}
    m["ann_return"] = annualised_return(portfolio_returns)
    m["ann_vol"] = annualised_volatility(portfolio_returns)
    m["sharpe"] = sharpe_ratio(portfolio_returns, rf=rf)
    m["max_drawdown"] = max_drawdown(portfolio_returns)
    m["calmar"] = calmar_ratio(portfolio_returns)

    if benchmark_returns is not None:
        m["information_ratio"] = information_ratio(portfolio_returns, benchmark_returns)

    if weights is not None:
        m["turnover"] = turnover(weights)

    if predictions is not None and actuals is not None:
        m["hit_rate"] = hit_rate(predictions, actuals)

    if regime_labels is not None:
        cond = regime_conditional_metrics(portfolio_returns, regime_labels, rf=rf)
        for regime, stats in cond.items():
            for stat_name, val in stats.items():
                m[f"regime_{regime}_{stat_name}"] = val

    return m
