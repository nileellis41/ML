"""Walk-forward validation engine.

Generates train/test folds with a strict no-leakage guarantee:
  - Training data ends at fold_end (exclusive)
  - Test data begins at fold_end (no overlap)
  - Refitting cadence is configurable per model class

Leakage invariant (enforced at generation time):
    max(train_dates) < min(test_dates)  for every fold

Use tests/test_walk_forward_no_leakage.py for an automated check.
"""
import logging
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Attributes
    ----------
    fold_id:
        Zero-based index of this fold.
    train_start:
        First date in the training window.
    train_end:
        Last date in the training window (inclusive).
    test_start:
        First date in the test window.
    test_end:
        Last date in the test window (inclusive).
    refit:
        Whether the model should be refit for this fold.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    refit: bool

    def __post_init__(self):
        if self.train_end >= self.test_start:
            raise ValueError(
                f"Fold {self.fold_id}: train_end ({self.train_end}) >= "
                f"test_start ({self.test_start}). This is a data leak."
            )


def generate_folds(
    dates: pd.DatetimeIndex,
    initial_train_years: int = 5,
    test_months: int = 1,
    refit_cadence_months: int = 12,
    expanding_window: bool = True,
) -> list[Fold]:
    """Generate walk-forward folds from a sorted DatetimeIndex.

    Parameters
    ----------
    dates:
        Sorted unique dates in the full dataset.
    initial_train_years:
        Length of the first training window in years.
    test_months:
        Length of each test fold in months.
    refit_cadence_months:
        How often to refit (in months). Folds between refits use the
        model fit at the prior refit date.
    expanding_window:
        If True, grow the training window each fold (expanding).
        If False, roll a fixed-length window.

    Returns
    -------
    list[Fold]
        Sequence of non-overlapping test folds with correct train/test splits.
    """
    dates = pd.DatetimeIndex(sorted(set(dates)))
    assert dates.is_monotonic_increasing, "dates must be sorted ascending"

    first_train_start = dates[0]
    initial_cutoff = first_train_start + pd.DateOffset(years=initial_train_years)

    # Snap to the nearest available date in the index
    train_end_loc = dates.searchsorted(initial_cutoff, side="right") - 1
    # Ensure the initial cutoff falls strictly inside the data and leaves at least
    # one test period after it.
    if train_end_loc < 1 or initial_cutoff > dates[-1]:
        raise ValueError(
            f"Insufficient data for initial_train_years={initial_train_years}. "
            f"Dates span {dates[0].date()} to {dates[-1].date()}."
        )

    folds: list[Fold] = []
    fold_id = 0
    months_since_refit = 0

    # Walk the test window forward month by month
    test_start_loc = train_end_loc + 1

    while test_start_loc < len(dates):
        test_start = dates[test_start_loc]
        test_end_approx = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        test_end_loc = min(
            dates.searchsorted(test_end_approx, side="right") - 1,
            len(dates) - 1,
        )
        test_end = dates[test_end_loc]

        train_end = dates[test_start_loc - 1]
        if expanding_window:
            current_train_start = first_train_start
        else:
            fixed_back = test_start - pd.DateOffset(years=initial_train_years)
            start_loc = max(0, dates.searchsorted(fixed_back, side="left"))
            current_train_start = dates[start_loc]

        refit = (months_since_refit == 0) or (months_since_refit >= refit_cadence_months)
        if refit:
            months_since_refit = 0

        fold = Fold(
            fold_id=fold_id,
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            refit=refit,
        )
        folds.append(fold)
        fold_id += 1
        months_since_refit += test_months
        test_start_loc = test_end_loc + 1

    logger.info(
        "Generated %d walk-forward folds (%s to %s); "
        "initial train=%d yr, test=%d mo, refit every %d mo",
        len(folds),
        folds[0].test_start.date() if folds else "N/A",
        folds[-1].test_end.date() if folds else "N/A",
        initial_train_years, test_months, refit_cadence_months,
    )
    return folds


def assert_no_leakage(folds: list[Fold], df: pd.DataFrame, date_col: str = "date") -> None:
    """Assert that no test-period rows appear in any training fold.

    Parameters
    ----------
    folds:
        List of Fold objects from generate_folds().
    df:
        Full dataset with a date column (or DatetimeIndex).
    date_col:
        Name of the date column if not the index.

    Raises
    ------
    AssertionError
        If any training slice contains dates from its corresponding test window.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    elif isinstance(df.index, pd.MultiIndex):
        # Panel with (date, ticker) index
        dates = df.index.get_level_values("date")
    else:
        dates = pd.DatetimeIndex(df[date_col])

    for fold in folds:
        train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
        train_dates = set(dates[train_mask])
        test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)
        test_dates = set(dates[test_mask])

        overlap = train_dates & test_dates
        assert not overlap, (
            f"DATA LEAK in fold {fold.fold_id}: {len(overlap)} dates appear in both "
            f"train [{fold.train_start.date()}, {fold.train_end.date()}] and "
            f"test [{fold.test_start.date()}, {fold.test_end.date()}].\n"
            f"Example overlap dates: {sorted(overlap)[:5]}"
        )

    logger.info("Leakage check PASSED: all %d folds are clean.", len(folds))


def split_fold(
    df: pd.DataFrame,
    fold: Fold,
    date_level: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice train and test DataFrames for a given fold.

    Works with both simple DatetimeIndex and (date, ticker) MultiIndex panels.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if isinstance(df.index, pd.MultiIndex):
        dates = df.index.get_level_values(date_level)
    else:
        dates = df.index

    train = df[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test = df[(dates >= fold.test_start) & (dates <= fold.test_end)]
    return train, test


def run_walk_forward(
    model,
    panel: pd.DataFrame,
    folds: list[Fold],
    feature_cols: list[str],
    target_col: str = "target",
) -> pd.DataFrame:
    """Execute walk-forward evaluation for a ModelInterface-compatible model.

    Parameters
    ----------
    model:
        Object implementing ModelInterface (fit / predict / get_params).
    panel:
        MultiIndex (date, ticker) panel with features and target.
    folds:
        Walk-forward folds from generate_folds().
    feature_cols:
        Column names to use as X.
    target_col:
        Column name to use as y.

    Returns
    -------
    pd.DataFrame
        Index = (date, ticker) of all test observations.
        Columns: 'prediction', 'target', 'fold_id'.
    """
    results: list[pd.DataFrame] = []
    fitted = False

    for fold in folds:
        train_df, test_df = split_fold(panel, fold)

        X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
        y_train = train_df[target_col].values
        X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)

        # Only drop rows where the target is unknown; features are zero-imputed above
        train_valid = ~np.isnan(y_train)
        test_valid = np.ones(len(X_test), dtype=bool)

        if fold.refit or not fitted:
            model.fit(X_train[train_valid], y_train[train_valid])
            fitted = True
            logger.info(
                "Fold %d: refit on %d train rows, testing on %d rows",
                fold.fold_id, train_valid.sum(), test_valid.sum(),
            )
        else:
            logger.debug(
                "Fold %d: skipping refit (cadence), testing on %d rows",
                fold.fold_id, test_valid.sum(),
            )

        preds = model.predict(X_test)

        result = test_df[[target_col]].copy()
        result["prediction"] = preds
        result["fold_id"] = fold.fold_id
        results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results).sort_index()
