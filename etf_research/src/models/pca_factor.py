"""PCA + Linear Factor Model.

Stage 1 (assumption modeling): PCA on macro feature panel extracts latent factors.
Stage 2 (coefficient modeling): OLS regresses ETF returns on the PCA factors.

PCA is fit on the training fold only -- never on test data.
Explained variance ratios and factor loadings are saved per fold for interpretation.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.models.base import ModelInterface

logger = logging.getLogger(__name__)


class PCAFactorModel(ModelInterface):
    """PCA-based latent factor model with OLS coefficient stage.

    Parameters
    ----------
    n_components:
        Number of PCA components (default 5).
    min_variance_explained:
        Minimum cumulative explained variance to accept (logged as warning if missed).
    fit_intercept:
        Whether OLS stage fits an intercept.
    """

    def __init__(
        self,
        n_components: int = 5,
        min_variance_explained: float = 0.80,
        fit_intercept: bool = True,
    ):
        self.n_components = n_components
        self.min_variance_explained = min_variance_explained
        self.fit_intercept = fit_intercept

        self._scaler = StandardScaler()
        self._pca: Optional[PCA] = None
        self._ols = LinearRegression(fit_intercept=fit_intercept)
        self._fitted = False

        # Logged per fold for downstream analysis
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_explained_: Optional[float] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit PCA on training data then OLS on PCA-reduced features."""
        X_scaled = self._scaler.fit_transform(X_train)

        n_comp = min(self.n_components, X_scaled.shape[1], X_scaled.shape[0])
        self._pca = PCA(n_components=n_comp)
        F_train = self._pca.fit_transform(X_scaled)

        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.cumulative_variance_explained_ = float(self.explained_variance_ratio_.cumsum()[-1])
        self.components_ = self._pca.components_

        if self.cumulative_variance_explained_ < self.min_variance_explained:
            logger.warning(
                "PCAFactorModel: cumulative variance explained=%.3f < threshold=%.3f "
                "with %d components. Consider increasing n_components.",
                self.cumulative_variance_explained_,
                self.min_variance_explained,
                n_comp,
            )
        else:
            logger.debug(
                "PCAFactorModel: %d components explain %.1f%% variance",
                n_comp, self.cumulative_variance_explained_ * 100,
            )

        self._ols.fit(F_train, y_train)
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted or self._pca is None:
            raise RuntimeError("PCAFactorModel has not been fitted yet.")
        X_scaled = self._scaler.transform(X_test)
        F_test = self._pca.transform(X_scaled)
        return self._ols.predict(F_test)

    def get_params(self) -> dict:
        return {
            "model": "PCAFactorModel",
            "n_components": self.n_components,
            "min_variance_explained": self.min_variance_explained,
            "cumulative_variance_explained": self.cumulative_variance_explained_,
            "fit_intercept": self.fit_intercept,
        }

    def get_factor_loadings(self, feature_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Return a DataFrame of factor loadings for interpretability analysis."""
        if self._pca is None:
            raise RuntimeError("Model not fitted.")
        n_comp = self._pca.n_components_
        index = feature_names or [f"feature_{i}" for i in range(self._pca.components_.shape[1])]
        cols = [f"PC{k+1}" for k in range(n_comp)]
        return pd.DataFrame(self._pca.components_.T, index=index, columns=cols)
