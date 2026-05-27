"""Deflated Sharpe Ratio — Bailey & López de Prado (2014).

Adjusts the observed Sharpe ratio for non-normality and the multiple-testing
inflation that arises when N configurations are evaluated and the best is
reported.

Functions
---------
probabilistic_sharpe_ratio  : P(SR > SR*)
deflated_sharpe_ratio       : PSR with benchmark = E[max SR over N trials]
min_track_record_length     : minimum T to achieve a target PSR

References
----------
Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio.
    Journal of Portfolio Management, 40(5), 94-107.
"""
import numpy as np
from scipy.stats import norm


_EULER_MASCHERONI = 0.5772156649


def _sharpe_se(
    observed_sr: float,
    n_obs: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    """Standard error of the annualised Sharpe estimator (BLP eq. 3).

    Parameters
    ----------
    observed_sr : annualised Sharpe ratio
    n_obs       : number of daily observations
    skew        : Fisher skewness of daily returns
    excess_kurt : Fisher (excess) kurtosis of daily returns  (0 for Gaussian)

    Notes
    -----
    BLP use γ₄ = Pearson kurtosis = excess_kurt + 3. The formula becomes:
        σ(SR̂) = sqrt((1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²) / (T-1))
               = sqrt((1 - skew·SR̂ + (excess_kurt+2)/4·SR̂²) / (T-1))
    """
    if n_obs <= 1:
        return np.nan
    numerator = 1.0 - skew * observed_sr + ((excess_kurt + 2) / 4.0) * observed_sr ** 2
    numerator = max(numerator, 1e-12)  # guard against imaginary SE
    return float(np.sqrt(numerator / (n_obs - 1)))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 0.0,
) -> float:
    """P(true SR > benchmark_sr) given observed Sharpe, sample size, and moments.

    Parameters
    ----------
    observed_sr  : annualised Sharpe ratio from the backtest
    benchmark_sr : minimum acceptable Sharpe (SR*)
    n_obs        : number of daily return observations
    skew         : Fisher skewness of daily returns
    kurt         : Fisher (excess) kurtosis of daily returns  (0 for Gaussian)

    Returns
    -------
    float in [0, 1]  — probability the strategy's true SR exceeds benchmark_sr.
    """
    se = _sharpe_se(observed_sr, n_obs, skew, kurt)
    if np.isnan(se) or se <= 0:
        return np.nan
    z = (observed_sr - benchmark_sr) / se
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurt: float = 0.0,
) -> float:
    """DSR = PSR(E[max SR over n_trials independent trials]).

    The DSR adjusts for the fact that reporting the best Sharpe out of N
    independent configurations inflates the expected maximum.

    Parameters
    ----------
    observed_sr  : annualised Sharpe ratio (best among n_trials)
    n_obs        : number of daily return observations
    n_trials     : number of independent configurations evaluated (N)
    sr_variance  : variance of the Sharpe estimator across trials;
                   use _sharpe_se(observed_sr, n_obs, skew, kurt)**2 when
                   the variance is not known from the trial distribution
    skew         : Fisher skewness of daily returns
    kurt         : Fisher (excess) kurtosis of daily returns

    Returns
    -------
    float in [0, 1]  — DSR p-value (probability the strategy is genuinely good)
    """
    if n_trials <= 0 or np.isnan(sr_variance) or sr_variance <= 0:
        return np.nan

    sigma_sr = np.sqrt(sr_variance)

    if n_trials == 1:
        # No specification search: benchmark = 0 (the null Sharpe)
        expected_max_sr = 0.0
    else:
        # Gumbel approximation for E[max of N iid N(0, sigma_sr²) RVs]
        g = _EULER_MASCHERONI
        z_n = (1 - g) * norm.ppf(1 - 1.0 / n_trials) + g * norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max_sr = sigma_sr * z_n

    return probabilistic_sharpe_ratio(observed_sr, expected_max_sr, n_obs, skew, kurt)


def min_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    prob_target: float = 0.95,
    skew: float = 0.0,
    kurt: float = 0.0,
) -> float:
    """Minimum number of daily observations to achieve PSR ≥ prob_target.

    Solves:
        T_min = 1 + (1 - skew·SR + (excess_kurt+2)/4·SR²) · (Φ⁻¹(p) / (SR - SR*))²

    Parameters
    ----------
    observed_sr  : observed annualised Sharpe ratio
    benchmark_sr : minimum acceptable Sharpe (SR*)
    prob_target  : target PSR (e.g. 0.95 for 95 % confidence)
    skew, kurt   : moments of daily returns

    Returns
    -------
    float  — minimum T in trading days (may be fractional; take ceil for integer).
    Returns np.inf if observed_sr <= benchmark_sr.
    """
    if observed_sr <= benchmark_sr:
        return np.inf

    z_target = float(norm.ppf(prob_target))
    numerator = 1.0 - skew * observed_sr + ((kurt + 2) / 4.0) * observed_sr ** 2
    numerator = max(numerator, 1e-12)
    sr_gap = observed_sr - benchmark_sr
    return float(1.0 + numerator * (z_target / sr_gap) ** 2)


def compute_dsr_for_returns(
    returns: "pd.Series",  # noqa: F821  (avoid top-level pandas import)
    n_trials: int = 8,
) -> float:
    """Compute DSR for a daily return series using observed return moments.

    Convenience wrapper: extracts n_obs, SR, skew, and excess_kurt from
    `returns`, computes sr_variance = _sharpe_se(...)², calls
    deflated_sharpe_ratio.

    Returns DSR p-value in [0, 1].
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    r = returns.dropna()
    n = len(r)
    if n < 20:
        return np.nan

    mu = float(r.mean())
    sig = float(r.std(ddof=1))
    if sig <= 0:
        return np.nan

    sr_ann = mu / sig * np.sqrt(252)
    skew = float(pd.Series(r).skew())
    excess_kurt = float(pd.Series(r).kurt())  # pandas .kurt() is Fisher (excess)

    se = _sharpe_se(sr_ann, n, skew, excess_kurt)
    if np.isnan(se) or se <= 0:
        return np.nan

    return deflated_sharpe_ratio(sr_ann, n, n_trials, se ** 2, skew, excess_kurt)
