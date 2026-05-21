"""LSTM return predictor.

Sequence-to-scalar LSTM predicting next-period log return.
Supports optional regime probability concatenation (_regime variant).
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import ModelInterface

logger = logging.getLogger(__name__)


class _LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        regime_dim: int = 0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        fc_input = hidden_size + regime_dim
        self.fc = nn.Linear(fc_input, 1)

    def forward(self, x: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = self.dropout(h_n[-1])  # last layer, all batches
        if regime is not None:
            h_last = torch.cat([h_last, regime], dim=1)
        return self.fc(h_last).squeeze(-1)


class LSTMReturnModel(ModelInterface):
    """LSTM return predictor.

    Parameters
    ----------
    seq_len:
        Number of lagged time steps fed to the LSTM (default 60).
    hidden_size:
        LSTM hidden state dimension.
    n_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout rate (applied between LSTM layers and before FC).
    lr:
        Adam learning rate.
    batch_size:
        Mini-batch size.
    epochs:
        Max training epochs.
    patience:
        Early-stopping patience (epochs without validation improvement).
    regime_dim:
        If > 0, the model appends a regime probability vector of this size
        to the LSTM hidden state before the output layer (_regime variant).
    device:
        'cuda', 'mps', or 'cpu'; auto-detected if None.
    """

    def __init__(
        self,
        seq_len: int = 60,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        regime_dim: int = 0,
        device: Optional[str] = None,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.regime_dim = regime_dim
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: Optional[_LSTMNet] = None
        self._fitted = False
        self._input_size: Optional[int] = None

    def _build_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        """Convert flat (T, F) feature matrix to (N, seq_len, F) sequences."""
        seqs, targets = [], []
        for i in range(self.seq_len, len(X) + 1):
            seqs.append(X[i - self.seq_len : i])
            if y is not None:
                targets.append(y[i - 1])
        if not seqs:
            return np.array([]), np.array([])
        seqs_arr = np.array(seqs, dtype=np.float32)
        if y is not None:
            return seqs_arr, np.array(targets, dtype=np.float32)
        return seqs_arr, None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_seq, y_seq = self._build_sequences(X_train, y_train)
        if len(X_seq) == 0:
            logger.warning("LSTMReturnModel: insufficient data for sequences. Skipping fit.")
            return

        self._input_size = X_seq.shape[2]
        self._net = _LSTMNet(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            regime_dim=self.regime_dim,
        ).to(self.device)

        # 90/10 train/val split for early stopping
        n_val = max(1, int(0.1 * len(X_seq)))
        X_tr, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_tr, y_val = y_seq[:-n_val], y_seq[-n_val:]

        dataset = TensorDataset(
            torch.from_numpy(X_tr).to(self.device),
            torch.from_numpy(y_tr).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

            # Validation
            self._net.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(self.device)
                yv = torch.from_numpy(y_val).to(self.device)
                val_loss = criterion(self._net(xv), yv).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                self._best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug("LSTM early stop at epoch %d (val_loss=%.6f)", epoch, best_val_loss)
                    break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self._net.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})
        self._fitted = True
        logger.debug("LSTMReturnModel fitted: %d samples, best_val_loss=%.6f", len(X_tr), best_val_loss)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted or self._net is None:
            raise RuntimeError("LSTMReturnModel has not been fitted.")
        X_seq, _ = self._build_sequences(X_test)
        if len(X_seq) == 0:
            return np.full(len(X_test), np.nan)

        self._net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_seq).to(self.device)
            preds = self._net(x_t).cpu().numpy()

        # Pad leading NaNs for sequence burn-in
        result = np.full(len(X_test), np.nan)
        result[self.seq_len - 1:] = preds
        return result

    def get_params(self) -> dict:
        return {
            "model": "LSTMReturnModel",
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "regime_dim": self.regime_dim,
        }
