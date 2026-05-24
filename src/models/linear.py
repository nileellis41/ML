"""Linear and logistic regression models.

LinearReturnModel  -- OLS predicting continuous next-period return magnitude.
LogisticDirectionModel -- sigmoid P(r_{t+1} > 0), one model per ETF.

Both are floor benchmarks: if LSTM cannot beat OLS here, something is wrong.
"""
import logging

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import ModelInterface

logger = logging.getLogger(__name__)


class LinearReturnModel(ModelInterface):
    """OLS regression predicting next-period log return.

    One instance per ETF.  Features are z-scored internally before fitting
    so coefficient magnitudes are comparable across folds.
    """

    def __init__(self, fit_intercept: bool = True, alpha: float = 1.0):
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        self._fitted = True
        logger.debug(
            "LinearReturnModel fit: %d samples, %d features, R²=%.4f",
            len(y_train), X_train.shape[1],
            self._model.score(X_scaled, y_train),
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.predict(self._scaler.transform(X_test))

    def get_params(self) -> dict:
        return {
            "model": "LinearReturnModel",
            "fit_intercept": self.fit_intercept,
            "alpha": self.alpha,
            "n_features": getattr(self._model, "n_features_in_", None),
        }


class LogisticDirectionModel(ModelInterface):
    """Logistic regression predicting P(r_{t+1} > 0).

    Output is a probability in [0, 1]; downstream allocator uses it as a
    directional conviction weight (complement to LinearReturnModel magnitude).
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, solver: str = "lbfgs"):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        binary = (y_train > 0).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            logger.warning("LogisticDirectionModel: degenerate target (all one class). Skipping fit.")
            self._fitted = False
            return
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, binary)
        self._fitted = True
        logger.debug(
            "LogisticDirectionModel fit: %d samples, up_rate=%.2f",
            len(binary), binary.mean(),
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return P(direction=up) for each row."""
        if not self._fitted:
            logger.warning("LogisticDirectionModel not fitted; returning 0.5 (neutral).")
            return np.full(len(X_test), 0.5)
        X_scaled = self._scaler.transform(X_test)
        return self._model.predict_proba(X_scaled)[:, 1]

    def get_params(self) -> dict:
        return {
            "model": "LogisticDirectionModel",
            "C": self.C,
            "max_iter": self.max_iter,
            "solver": self.solver,
        }
