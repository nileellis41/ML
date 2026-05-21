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
        results[str(label)] = {
            "ann_return": annualised_return(r_slice),
            "ann_vol": annualised_volatility(r_slice),
            "sharpe": sharpe_ratio(r_slice, rf=rf),
            "max_drawdown": max_drawdown(r_slice),
            "n_periods": int(mask.sum()),
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
