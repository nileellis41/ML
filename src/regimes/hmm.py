"""Gaussian HMM regime detector.

4-state Gaussian HMM fit on (SPY returns, VIX, HY OAS, term spread).
States are labeled post-hoc by mean return + vol characteristics.
Emits out-of-sample state probabilities only -- no leakage.

Output: data/regimes/hmm_probs.parquet
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.utils.io import save_parquet

logger = logging.getLogger(__name__)

_N_STATES = 4
_STATE_LABELS = ["risk_on", "neutral", "risk_off", "crisis"]
_OUTPUT_DIR = Path("data/regimes")


def _build_hmm_inputs(
    spy_close: pd.Series,
    vix: pd.Series,
    hy_oas: pd.Series,
    term_spread: pd.Series,
) -> pd.DataFrame:
    """Assemble and z-score the 4 HMM input series.

    Uses 252-day rolling z-scores to keep features stationary without
    look-ahead (z-score parameters computed from trailing data only).
    """
    spy_ret = np.log(spy_close / spy_close.shift(1))

    features = pd.DataFrame({
        "spy_ret": spy_ret,
        "vix": vix,
        "hy_oas": hy_oas,
        "term_spread": term_spread,
    }).dropna()

    # Rolling z-score (252-day window, min 60 periods)
    zscored = (features - features.rolling(252, min_periods=60).mean()) / \
               features.rolling(252, min_periods=60).std()
    return zscored.dropna()


def _label_states(probs: np.ndarray, spy_returns: np.ndarray) -> dict[int, str]:
    """Assign interpretable labels to HMM states by mean return + volatility.

    States are sorted by (mean_return DESC, vol ASC) and mapped to labels:
    risk_on > neutral > risk_off > crisis.
    """
    state_seq = probs.argmax(axis=1)
    stats = {}
    for s in range(_N_STATES):
        mask = state_seq == s
        if mask.sum() == 0:
            stats[s] = (0.0, float("inf"))
        else:
            stats[s] = (spy_returns[mask].mean(), spy_returns[mask].std())

    # Sort by mean_return desc, then vol asc (proxy for crisis)
    sorted_states = sorted(stats.keys(), key=lambda s: (-stats[s][0], stats[s][1]))
    label_map = {state: label for state, label in zip(sorted_states, _STATE_LABELS)}
    return label_map


def fit_hmm_rolling(
    spy_close: pd.Series,
    vix: pd.Series,
    hy_oas: pd.Series,
    term_spread: pd.Series,
    initial_train_years: int = 5,
    n_states: int = _N_STATES,
    n_iter: int = 100,
    covariance_type: str = "full",
    save: bool = True,
    version: str = "v1",
) -> pd.DataFrame:
    """Fit HMM with rolling walk-forward to emit out-of-sample state probabilities.

    Parameters
    ----------
    spy_close, vix, hy_oas, term_spread:
        Daily time series (DatetimeIndex, same calendar).
    initial_train_years:
        Length of first training window.
    n_states:
        Number of hidden states (4 by default).
    n_iter:
        EM algorithm iterations.
    covariance_type:
        HMM covariance structure ('full', 'diag', 'tied', 'spherical').
    save:
        Write output to data/regimes/hmm_probs.parquet.
    version:
        Version string stored as metadata column.

    Returns
    -------
    pd.DataFrame
        Columns: state_0, state_1, state_2, state_3 (posterior probabilities),
                 predicted_state (argmax), predicted_label, version.
        Index: date (only out-of-sample dates).
    """
    features = _build_hmm_inputs(spy_close, vix, hy_oas, term_spread)
    spy_ret = np.log(spy_close / spy_close.shift(1)).reindex(features.index)

    all_probs: list[pd.DataFrame] = []

    train_end = features.index[0] + pd.DateOffset(years=initial_train_years)
    train_end_loc = features.index.searchsorted(train_end, side="right") - 1

    if train_end_loc < n_states * 2:
        raise ValueError("Insufficient data for HMM initial training window.")

    # Walk forward: expand training window by 1 month at a time
    step_months = 1
    test_start_loc = train_end_loc + 1

    while test_start_loc < len(features):
        test_end = features.index[test_start_loc] + pd.DateOffset(months=step_months)
        test_end_loc = min(
            features.index.searchsorted(test_end, side="right") - 1,
            len(features) - 1,
        )

        X_train = features.iloc[:test_start_loc].values
        X_test = features.iloc[test_start_loc : test_end_loc + 1].values

        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=42,
            )
            model.fit(X_train)
            posteriors = model.predict_proba(X_test)
        except Exception as exc:
            logger.warning(
                "HMM fit failed at step starting %s: %s. Filling with uniform probs.",
                features.index[test_start_loc].date(), exc,
            )
            posteriors = np.full((len(X_test), n_states), 1.0 / n_states)

        test_dates = features.index[test_start_loc : test_end_loc + 1]
        test_spy = spy_ret.reindex(test_dates).values

        label_map = _label_states(posteriors, test_spy)

        df = pd.DataFrame(
            posteriors,
            index=test_dates,
            columns=[f"state_{i}" for i in range(n_states)],
        )
        df["predicted_state"] = posteriors.argmax(axis=1)
        df["predicted_label"] = df["predicted_state"].map(label_map)
        df["version"] = version
        all_probs.append(df)

        test_start_loc = test_end_loc + 1

    if not all_probs:
        raise RuntimeError("HMM produced no out-of-sample probabilities.")

    result = pd.concat(all_probs).sort_index()
    result.index.name = "date"

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_parquet(result, _OUTPUT_DIR / "hmm_probs.parquet")
        logger.info("HMM regime probabilities saved: %d dates", len(result))

    return result
