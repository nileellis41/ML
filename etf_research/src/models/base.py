"""Abstract ModelInterface that all prediction models must implement."""
from abc import ABC, abstractmethod

import numpy as np


class ModelInterface(ABC):
    """Common interface for all return prediction models."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data.

        Parameters
        ----------
        X_train:
            Feature matrix, shape (n_samples, n_features).
        y_train:
            Target vector (forward returns), shape (n_samples,).
        """

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted values for X_test.

        Parameters
        ----------
        X_test:
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,).
        """

    @abstractmethod
    def get_params(self) -> dict:
        """Return hyperparameters and metadata for logging."""
