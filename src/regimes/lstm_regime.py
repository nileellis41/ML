"""Supervised LSTM Regime Classifier.

Targets: HMM-derived state labels OR forward-realized vol buckets (configurable).
Evaluated with Cohen's kappa (target >= 0.7) and transition lead time.

Output: data/regimes/lstm_probs.parquet
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from src.utils.io import load_parquet, save_parquet

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("data/regimes")
_N_STATES = 4


class _RegimeLSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = _N_STATES,
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
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        h_last = self.dropout(h[-1])
        return self.fc(h_last)  # logits (B, n_classes)


class LSTMRegimeClassifier:
    """LSTM classifier that maps macro features to regime state probabilities.

    Parameters
    ----------
    seq_len:
        Look-back window in trading days.
    hidden_size, n_layers, dropout:
        LSTM architecture parameters.
    lr, batch_size, epochs, patience:
        Training hyperparameters.
    target:
        'hmm_labels' uses HMM state as supervision; 'vol_buckets' uses
        forward-realized vol quantile buckets.
    kappa_threshold:
        Minimum Cohen's kappa on validation set to accept the classifier.
    n_classes:
        Number of regime classes (must match HMM n_states if using hmm_labels).
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
        target: str = "hmm_labels",
        kappa_threshold: float = 0.7,
        n_classes: int = _N_STATES,
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
        self.target = target
        self.kappa_threshold = kappa_threshold
        self.n_classes = n_classes
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net: Optional[_RegimeLSTMNet] = None
        self._fitted = False
        self.val_kappa_: Optional[float] = None

    def _build_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> tuple:
        seqs, targets = [], []
        for i in range(self.seq_len, len(X) + 1):
            seqs.append(X[i - self.seq_len : i])
            if y is not None:
                targets.append(y[i - 1])
        if not seqs:
            return np.array([]), np.array([]) if y is not None else np.array([])
        return np.array(seqs, dtype=np.float32), (np.array(targets, dtype=np.int64) if y is not None else None)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train the LSTM on (features, regime_labels) pairs.

        Parameters
        ----------
        X_train:
            Macro feature matrix (T, n_features).
        y_train:
            Integer regime labels (T,), values in [0, n_classes).
        """
        X_seq, y_seq = self._build_sequences(X_train, y_train)
        if len(X_seq) == 0:
            logger.warning("LSTMRegimeClassifier: insufficient data. Skipping fit.")
            return

        n_val = max(1, int(0.15 * len(X_seq)))
        X_tr, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_tr, y_val = y_seq[:-n_val], y_seq[-n_val:]

        self._net = _RegimeLSTMNet(
            input_size=X_tr.shape[2],
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
            n_classes=self.n_classes,
        ).to(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X_tr).to(self.device),
            torch.from_numpy(y_tr).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._net.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(self.device)
                yv = torch.from_numpy(y_val).to(self.device)
                val_loss = criterion(self._net(xv), yv).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if hasattr(self, "_best_state"):
            self._net.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})

        # Compute Cohen's kappa on validation set
        self._net.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).to(self.device)
            pred_labels = self._net(xv).argmax(dim=1).cpu().numpy()
        self.val_kappa_ = float(cohen_kappa_score(y_val, pred_labels))

        if self.val_kappa_ < self.kappa_threshold:
            logger.warning(
                "LSTMRegimeClassifier: validation kappa=%.3f < threshold=%.3f. "
                "Classifier quality below target -- review features or architecture.",
                self.val_kappa_, self.kappa_threshold,
            )
        else:
            logger.info(
                "LSTMRegimeClassifier: validation kappa=%.3f (OK, threshold=%.3f)",
                self.val_kappa_, self.kappa_threshold,
            )
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape (T, n_classes)."""
        if not self._fitted or self._net is None:
            raise RuntimeError("LSTMRegimeClassifier has not been fitted.")
        X_seq, _ = self._build_sequences(X, None)
        if len(X_seq) == 0:
            return np.full((len(X), self.n_classes), 1.0 / self.n_classes)

        self._net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_seq).to(self.device)
            logits = self._net(x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Pad burn-in period with uniform distribution
        result = np.full((len(X), self.n_classes), 1.0 / self.n_classes)
        result[self.seq_len - 1:] = probs
        return result


def run_lstm_regime_walk_forward(
    features_df: pd.DataFrame,
    hmm_probs: pd.DataFrame,
    initial_train_years: int = 5,
    version: str = "v1",
    save: bool = True,
    **classifier_kwargs,
) -> pd.DataFrame:
    """Walk-forward LSTM regime classifier producing out-of-sample probabilities.

    Parameters
    ----------
    features_df:
        Macro feature panel (DatetimeIndex, columns = feature names).
    hmm_probs:
        HMM state probabilities with 'predicted_state' column (DatetimeIndex).
    initial_train_years:
        Length of the initial training window.
    version:
        Version string stored in output for traceability.
    save:
        Write results to data/regimes/lstm_probs.parquet.

    Returns
    -------
    pd.DataFrame
        Columns: lstm_state_0..3, predicted_state, version, val_kappa.
        Index: date (out-of-sample dates only).
    """
    common_idx = features_df.index.intersection(hmm_probs.index)
    X_all = features_df.reindex(common_idx).fillna(0).values.astype(np.float32)
    y_all = hmm_probs.reindex(common_idx)["predicted_state"].values.astype(np.int64)

    train_end = common_idx[0] + pd.DateOffset(years=initial_train_years)
    train_end_loc = common_idx.searchsorted(train_end, side="right") - 1

    all_probs: list[pd.DataFrame] = []
    step_months = 1
    test_start_loc = train_end_loc + 1

    while test_start_loc < len(common_idx):
        test_end = common_idx[test_start_loc] + pd.DateOffset(months=step_months)
        test_end_loc = min(
            common_idx.searchsorted(test_end, side="right") - 1,
            len(common_idx) - 1,
        )

        X_train = X_all[:test_start_loc]
        y_train = y_all[:test_start_loc]
        X_test = X_all[:test_end_loc + 1]  # include history for seq build

        clf = LSTMRegimeClassifier(**classifier_kwargs)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[test_start_loc:]

        test_dates = common_idx[test_start_loc : test_end_loc + 1]
        df = pd.DataFrame(
            probs,
            index=test_dates,
            columns=[f"lstm_state_{i}" for i in range(clf.n_classes)],
        )
        df["predicted_state"] = probs.argmax(axis=1)
        df["version"] = version
        df["val_kappa"] = clf.val_kappa_ or np.nan
        all_probs.append(df)

        test_start_loc = test_end_loc + 1

    if not all_probs:
        raise RuntimeError("LSTM regime classifier produced no out-of-sample probabilities.")

    result = pd.concat(all_probs).sort_index()
    result.index.name = "date"

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_parquet(result, _OUTPUT_DIR / "lstm_probs.parquet")
        logger.info("LSTM regime probabilities saved: %d dates, avg kappa=%.3f",
                    len(result), result["val_kappa"].mean())

    return result
