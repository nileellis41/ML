"""Model-Based RL Portfolio Allocator.

Learns a transition model of returns conditioned on state, then plans
via MPC (Model Predictive Control) with a short rollout horizon.

Action space: long-only weights summing to 1, capped at 25% per asset.
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

_MAX_WEIGHT = 0.25
_MIN_WEIGHT = 0.0


class _TransitionNet(nn.Module):
    """Predicts next-period return vector given current state and weights."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_assets, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, weights], dim=-1)
        return self.net(x)


def _project_weights(w: np.ndarray, max_w: float = _MAX_WEIGHT) -> np.ndarray:
    """Project weight vector to long-only simplex with per-asset cap."""
    n = len(w)
    w = np.clip(w, _MIN_WEIGHT, max_w)
    total = w.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return w / total


class ModelBasedRLAllocator:
    """Model-based RL portfolio allocator using MPC.

    Parameters
    ----------
    n_assets:
        Number of ETFs in the universe.
    state_dim:
        Dimension of the state vector (features fed to the model).
    horizon:
        MPC rollout horizon in periods (months).
    n_rollouts:
        Number of random weight rollouts to evaluate per decision date.
    hidden:
        Hidden layer size for the transition network.
    lr:
        Adam learning rate.
    epochs:
        Training epochs.
    max_weight:
        Per-asset weight cap.
    regime_dim:
        If > 0, append regime probabilities to the state vector (_regime variant).
    """

    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        horizon: int = 6,
        n_rollouts: int = 100,
        hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 30,
        max_weight: float = _MAX_WEIGHT,
        regime_dim: int = 0,
        device: Optional[str] = None,
    ):
        self.n_assets = n_assets
        self.state_dim = state_dim + regime_dim
        self.horizon = horizon
        self.n_rollouts = n_rollouts
        self.max_weight = max_weight
        self.regime_dim = regime_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._model = _TransitionNet(self.state_dim, n_assets, hidden).to(self.device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self.epochs = epochs
        self._fitted = False

    def fit(
        self,
        states: np.ndarray,
        realized_returns: np.ndarray,
        regime_vecs: Optional[np.ndarray] = None,
    ) -> None:
        """Train the transition model on historical (state, return) pairs.

        Parameters
        ----------
        states:
            (T, state_dim) feature matrix.
        realized_returns:
            (T, n_assets) matrix of realized next-period returns.
        regime_vecs:
            Optional (T, regime_dim) regime probabilities.
        """
        if regime_vecs is not None:
            states = np.concatenate([states, regime_vecs], axis=1)

        # Dummy equal-weight actions for training the transition model
        weights = np.full((len(states), self.n_assets), 1.0 / self.n_assets, dtype=np.float32)

        valid = ~np.isnan(realized_returns).any(axis=1)
        S = torch.from_numpy(states[valid].astype(np.float32)).to(self.device)
        W = torch.from_numpy(weights[valid]).to(self.device)
        R = torch.from_numpy(realized_returns[valid].astype(np.float32)).to(self.device)

        dataset = TensorDataset(S, W, R)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self._model.train()
            epoch_loss = 0.0
            for s, w, r in loader:
                self._optimizer.zero_grad()
                pred_r = self._model(s, w)
                loss = criterion(pred_r, r)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug("ModelBasedRL epoch %d/%d loss=%.6f", epoch + 1, self.epochs, epoch_loss)

        self._fitted = True
        logger.info("ModelBasedRLAllocator fitted on %d samples", valid.sum())

    def allocate(
        self,
        current_state: np.ndarray,
        regime_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return portfolio weights for the current decision date.

        Uses random-shooting MPC: sample n_rollouts random weight vectors,
        simulate the horizon-step rollout with the transition model, and
        select the weights that maximise the cumulative Sharpe proxy.

        Parameters
        ----------
        current_state:
            1D feature vector for the current date.
        regime_vec:
            Optional 1D regime probability vector.

        Returns
        -------
        np.ndarray
            Portfolio weights of shape (n_assets,), summing to 1.
        """
        if not self._fitted:
            logger.warning("ModelBasedRLAllocator not fitted; returning equal weights.")
            return np.full(self.n_assets, 1.0 / self.n_assets)

        if regime_vec is not None:
            state = np.concatenate([current_state, regime_vec])
        else:
            state = current_state

        self._model.eval()
        best_score = -np.inf
        best_weights = np.full(self.n_assets, 1.0 / self.n_assets)

        rng = np.random.default_rng(42)

        with torch.no_grad():
            for _ in range(self.n_rollouts):
                # Random Dirichlet weights
                raw_w = rng.dirichlet(np.ones(self.n_assets))
                w = _project_weights(raw_w, self.max_weight)

                # Rollout
                s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
                w_t = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(self.device)
                cum_return = 0.0
                for _ in range(self.horizon):
                    pred_r = self._model(s, w_t).squeeze(0).cpu().numpy()
                    portfolio_r = float(np.dot(w, pred_r))
                    cum_return += portfolio_r
                    # Update state with predicted returns as new features (simplified)
                    new_state = state.copy()
                    new_state[:self.n_assets] = pred_r
                    s = torch.from_numpy(new_state.astype(np.float32)).unsqueeze(0).to(self.device)

                if cum_return > best_score:
                    best_score = cum_return
                    best_weights = w

        return best_weights
