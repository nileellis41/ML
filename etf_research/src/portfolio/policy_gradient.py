"""Policy Gradient Portfolio Allocator (PPO).

State: recent returns + volatility + regime probabilities (if _regime variant).
Action: portfolio weight vector via softmax.
Reward: differential Sharpe ratio (Moody & Saffell) minus turnover penalty.

Uses a custom gym-compatible environment to stay independent of stable-baselines3
version churn; PPO is implemented in PyTorch directly.
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_MAX_WEIGHT = 0.25
_MIN_WEIGHT = 0.0


def _project_weights_torch(w: torch.Tensor, max_w: float = _MAX_WEIGHT) -> torch.Tensor:
    w = torch.clamp(w, _MIN_WEIGHT, max_w)
    s = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return w / s


class _ActorCriticNet(nn.Module):
    """Shared trunk with separate actor (policy) and critic (value) heads."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Actor outputs logits for softmax weights
        self.actor = nn.Linear(hidden, n_assets)
        # Critic outputs scalar state value
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_weights(self, state: torch.Tensor, max_w: float = _MAX_WEIGHT) -> torch.Tensor:
        logits, _ = self(state)
        raw = F.softmax(logits, dim=-1)
        return _project_weights_torch(raw, max_w)


def _differential_sharpe(returns: list[float], eta: float = 0.01) -> list[float]:
    """Online differential Sharpe ratio rewards (Moody & Saffell, 1998).

    Approximates dS/dw_t by exponential moving average of return moments.
    """
    rewards = []
    A, B = 0.0, 0.0
    for r in returns:
        dA = r - A
        dB = r ** 2 - B
        D = B - A ** 2
        reward = (B * dA - 0.5 * A * dB) / (D ** 1.5 + 1e-8) if D > 1e-8 else 0.0
        A += eta * dA
        B += eta * dB
        rewards.append(reward)
    return rewards


class PPOAllocator:
    """Custom PPO portfolio allocator.

    Parameters
    ----------
    n_assets:
        Number of ETFs in the universe.
    state_dim:
        Dimension of the state vector.
    hidden:
        Hidden layer size.
    lr:
        Adam learning rate.
    gamma:
        Discount factor.
    clip_eps:
        PPO clipping epsilon.
    n_epochs:
        PPO epochs per update.
    batch_size:
        Mini-batch size for PPO updates.
    total_timesteps:
        Total environment steps.
    turnover_penalty:
        Per-period turnover cost (fraction of rebalanced notional).
    max_weight:
        Per-asset weight cap.
    regime_dim:
        If > 0, regime probability vector is appended to state (_regime variant).
    """

    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        hidden: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        total_timesteps: int = 100_000,
        turnover_penalty: float = 0.001,
        max_weight: float = _MAX_WEIGHT,
        regime_dim: int = 0,
        device: Optional[str] = None,
    ):
        self.n_assets = n_assets
        self.state_dim = state_dim + regime_dim
        self.regime_dim = regime_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.total_timesteps = total_timesteps
        self.turnover_penalty = turnover_penalty
        self.max_weight = max_weight
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._net = _ActorCriticNet(self.state_dim, n_assets, hidden).to(self.device)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._fitted = False

    def _make_state(
        self,
        features: np.ndarray,
        regime_vec: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if regime_vec is not None:
            s = np.concatenate([features, regime_vec])
        else:
            s = features
        return torch.from_numpy(s.astype(np.float32)).unsqueeze(0).to(self.device)

    def fit(
        self,
        states: np.ndarray,
        realized_returns: np.ndarray,
        regime_vecs: Optional[np.ndarray] = None,
    ) -> None:
        """Train the PPO policy on historical (state, return) trajectory.

        Parameters
        ----------
        states:
            (T, state_dim) feature matrix.
        realized_returns:
            (T, n_assets) matrix of realised returns per period.
        regime_vecs:
            Optional (T, regime_dim) array.
        """
        if regime_vecs is not None:
            all_states = np.concatenate([states, regime_vecs], axis=1).astype(np.float32)
        else:
            all_states = states.astype(np.float32)

        T = len(all_states)
        valid = ~np.isnan(realized_returns).any(axis=1)
        s_arr = all_states[valid]
        r_arr = realized_returns[valid].astype(np.float32)

        prev_weights = np.full(self.n_assets, 1.0 / self.n_assets)
        trajectory = {"states": [], "actions": [], "log_probs": [], "rewards": [], "values": []}

        # Collect one full trajectory pass through history
        self._net.eval()
        with torch.no_grad():
            for t in range(len(s_arr)):
                st = torch.from_numpy(s_arr[t]).unsqueeze(0).to(self.device)
                logits, value = self._net(st)
                raw = F.softmax(logits, dim=-1)
                weights = _project_weights_torch(raw, self.max_weight).squeeze(0).cpu().numpy()

                port_ret = float(np.dot(weights, r_arr[t]))
                turnover = float(np.abs(weights - prev_weights).sum())
                reward = port_ret - self.turnover_penalty * turnover

                log_prob = F.log_softmax(logits, dim=-1).gather(
                    1, torch.from_numpy(weights.argmax(keepdims=True)[np.newaxis]).to(self.device)
                ).item()

                trajectory["states"].append(s_arr[t])
                trajectory["actions"].append(weights)
                trajectory["log_probs"].append(log_prob)
                trajectory["rewards"].append(reward)
                trajectory["values"].append(value.item())
                prev_weights = weights

        # Compute discounted returns
        G = 0.0
        returns = []
        for r in reversed(trajectory["rewards"]):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values_t = torch.tensor(trajectory["values"], dtype=torch.float32).to(self.device)
        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.from_numpy(np.array(trajectory["states"])).to(self.device)
        old_log_probs_t = torch.tensor(trajectory["log_probs"]).to(self.device)

        # PPO update epochs
        self._net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(len(states_t))
            for start in range(0, len(states_t), self.batch_size):
                idx = perm[start : start + self.batch_size]
                sb = states_t[idx]
                old_lp = old_log_probs_t[idx]
                adv = advantages[idx]
                ret = returns_t[idx]

                logits, values = self._net(sb)
                log_probs = F.log_softmax(logits, dim=-1).mean(dim=-1)

                ratio = torch.exp(log_probs - old_lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, ret)
                loss = actor_loss + 0.5 * critic_loss

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 0.5)
                self._optimizer.step()

        self._fitted = True
        total_reward = sum(trajectory["rewards"])
        logger.info(
            "PPOAllocator fitted: %d steps, total_reward=%.4f", len(s_arr), total_reward
        )

    def allocate(
        self,
        current_state: np.ndarray,
        regime_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return portfolio weights for the current decision date."""
        if not self._fitted:
            logger.warning("PPOAllocator not fitted; returning equal weights.")
            return np.full(self.n_assets, 1.0 / self.n_assets)

        st = self._make_state(current_state, regime_vec)
        self._net.eval()
        with torch.no_grad():
            weights = self._net.get_weights(st, self.max_weight).squeeze(0).cpu().numpy()
        return weights
