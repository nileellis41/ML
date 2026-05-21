"""Vector Autoregression (VAR) model.

Captures cross-ETF dynamics.  BIC-selected lag order (capped at 5).
Wraps statsmodels VARModel behind the ModelInterface.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from src.models.base import ModelInterface

logger = logging.getLogger(__name__)

_MAX_LAGS = 5


class VARReturnModel(ModelInterface):
    """VAR model predicting next-period returns for all ETFs jointly.

    Parameters
    ----------
    maxlags:
        Maximum lag order to consider; BIC selects the optimal value up to this cap.
    asset_cols:
        Column names corresponding to return series for each ETF.
        If None, all columns are treated as endogenous.
    """

    def __init__(
        self,
        maxlags: int = _MAX_LAGS,
        asset_cols: Optional[list[str]] = None,
    ):
        self.maxlags = maxlags
        self.asset_cols = asset_cols
        self._fitted_result = None
        self._selected_lag: Optional[int] = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit VAR on the multi-asset return matrix.

        X_train is treated as a (T, n_assets) matrix of historical returns.
        y_train is ignored here (VAR is self-supervised on the return panel).
        """
        df = pd.DataFrame(X_train)
        var_model = VAR(df)
        try:
            selection = var_model.select_order(maxlags=self.maxlags)
            selected = selection.selected_orders.get("bic", 1)
            self._selected_lag = max(1, selected)
        except Exception:
            self._selected_lag = 1

        self._fitted_result = var_model.fit(maxlags=self._selected_lag)
        self._fitted = True
        logger.debug(
            "VARReturnModel: BIC-selected lag=%d, fitted on %d obs x %d series",
            self._selected_lag, len(X_train), X_train.shape[1],
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return 1-step-ahead forecast for each row in X_test.

        For walk-forward use: each row of X_test is treated as the most recent
        lag window.  Returns the predicted values for each asset column.
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("VARReturnModel has not been fitted.")

        lag = self._selected_lag or 1
        # Use last `lag` rows as the conditioning history
        history = X_test[-lag:] if len(X_test) >= lag else X_test
        forecast = self._fitted_result.forecast(y=history, steps=1)  # (1, n_assets)
        return forecast.flatten()

    def predict_sequence(self, X_test: np.ndarray) -> np.ndarray:
        """Predict one step ahead for each overlapping window in X_test.

        Suitable for walk-forward where each row is the last observation
        available at the decision date.
        """
        if not self._fitted or self._fitted_result is None:
            raise RuntimeError("VARReturnModel has not been fitted.")

        lag = self._selected_lag or 1
        preds = []
        for i in range(len(X_test)):
            start = max(0, i - lag + 1)
            window = X_test[start : i + 1]
            if len(window) < lag:
                preds.append(np.full(X_test.shape[1], np.nan))
                continue
            fc = self._fitted_result.forecast(y=window, steps=1)
            preds.append(fc.flatten())
        return np.array(preds)

    def get_params(self) -> dict:
        return {
            "model": "VARReturnModel",
            "maxlags": self.maxlags,
            "selected_lag": self._selected_lag,
        }
