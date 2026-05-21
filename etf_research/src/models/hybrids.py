"""Hybrid models: VAR+LSTM, PCA+LSTM, CNN+LSTM.

VAR_LSTM  : VAR predicts linear component; LSTM learns residuals.
PCA_LSTM  : PCA components fed to LSTM as a low-dimensional input sequence.
CNN_LSTM  : CNN encodes each candlestick image; LSTM consumes the sequence.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import ModelInterface
from src.models.lstm import LSTMReturnModel, _LSTMNet
from src.models.var import VARReturnModel
from src.models.pca_factor import PCAFactorModel
from src.models.cnn import CNNCandlestickModel, _CNNNet

logger = logging.getLogger(__name__)


class VARLSTMModel(ModelInterface):
    """VAR for linear component + LSTM on residuals.

    Parameters
    ----------
    lstm_params:
        Keyword arguments forwarded to LSTMReturnModel.
    var_maxlags:
        Maximum lag order for the VAR component.
    """

    def __init__(self, var_maxlags: int = 5, **lstm_params):
        self.var_maxlags = var_maxlags
        self._var = VARReturnModel(maxlags=var_maxlags)
        self._lstm = LSTMReturnModel(**lstm_params)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # VAR on the return features to get linear predictions
        self._var.fit(X_train, y_train)
        var_preds = self._var.predict_sequence(X_train)

        # LSTM on residuals (y - VAR_prediction)
        if var_preds.ndim > 1:
            var_preds_1d = var_preds[:, 0]  # first asset column
        else:
            var_preds_1d = var_preds

        residuals = y_train - var_preds_1d
        valid = ~np.isnan(residuals)
        self._lstm.fit(X_train[valid], residuals[valid])
        self._var_preds_train_ = var_preds_1d
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("VARLSTMModel has not been fitted.")
        var_preds = self._var.predict_sequence(X_test)
        if var_preds.ndim > 1:
            var_preds_1d = var_preds[:, 0]
        else:
            var_preds_1d = var_preds
        lstm_resid = self._lstm.predict(X_test)
        return var_preds_1d + np.nan_to_num(lstm_resid, nan=0.0)

    def get_params(self) -> dict:
        return {
            "model": "VARLSTMModel",
            "var_maxlags": self.var_maxlags,
            **self._lstm.get_params(),
        }


class PCALSTMModel(ModelInterface):
    """PCA for dimensionality reduction; LSTM on the factor scores.

    PCA is fit on training data only (no leakage).  The K-dimensional
    factor score sequence is then fed to an LSTM.
    """

    def __init__(
        self,
        n_pca_components: int = 5,
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
        self.n_pca_components = n_pca_components
        self._pca_model = PCAFactorModel(n_components=n_pca_components)
        self._lstm = LSTMReturnModel(
            seq_len=seq_len, hidden_size=hidden_size, n_layers=n_layers,
            dropout=dropout, lr=lr, batch_size=batch_size,
            epochs=epochs, patience=patience, regime_dim=regime_dim,
            device=device,
        )
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        self._scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        n_comp = min(self.n_pca_components, X_scaled.shape[1])
        self._pca = PCA(n_components=n_comp)
        F_train = self._pca.fit_transform(X_scaled)
        logger.debug(
            "PCALSTMModel: PCA explains %.1f%% variance with %d components",
            self._pca.explained_variance_ratio_.sum() * 100, n_comp,
        )
        self._lstm.fit(F_train, y_train)
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCALSTMModel has not been fitted.")
        X_scaled = self._scaler.transform(X_test)
        F_test = self._pca.transform(X_scaled)
        return self._lstm.predict(F_test)

    def get_params(self) -> dict:
        return {
            "model": "PCALSTMModel",
            "n_pca_components": self.n_pca_components,
            **self._lstm.get_params(),
        }


class CNNLSTMModel:
    """CNN encodes candlestick images; LSTM consumes the feature sequence.

    For each decision date, we generate a sequence of N overlapping
    60-day images (stepped weekly), encode each with the CNN backbone,
    and feed the (N, 128) sequence to an LSTM for the final forecast.

    Parameters
    ----------
    n_images:
        Number of overlapping 60-day windows per decision date (default 12).
    step_days:
        Step size in trading days between consecutive windows (default 5).
    lstm_hidden:
        LSTM hidden size.
    regime_dim:
        Regime probability dimensions appended to LSTM output.
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_images: int = 12,
        step_days: int = 5,
        lstm_hidden: int = 64,
        cnn_dropout: float = 0.5,
        lstm_dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 16,
        epochs: int = 50,
        patience: int = 10,
        regime_dim: int = 0,
        device: Optional[str] = None,
        candle_cache_dir: Path = Path("data/processed/candles"),
    ):
        self.in_channels = in_channels
        self.n_images = n_images
        self.step_days = step_days
        self.lstm_hidden = lstm_hidden
        self.regime_dim = regime_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.candle_cache_dir = Path(candle_cache_dir)

        self._cnn_backbone = _CNNNet(in_channels=in_channels, dropout=cnn_dropout, regime_dim=0).to(self.device)
        # Remove the FC head -- use CNN as a 128-dim feature extractor
        self._cnn_backbone.fc = nn.Identity()

        lstm_in = 128
        self._lstm_head = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=lstm_dropout,
            batch_first=True,
        ).to(self.device)
        self._dropout = nn.Dropout(lstm_dropout).to(self.device)
        fc_in = lstm_hidden + regime_dim
        self._output_fc = nn.Linear(fc_in, 1).to(self.device)
        self._fitted = False

    def _encode_sequence(self, image_path_seq: list[list[Path]]) -> np.ndarray:
        """Encode a list of image sequences through the CNN backbone.

        Parameters
        ----------
        image_path_seq:
            List of length N_samples, each element a list of n_images Path objects.

        Returns
        -------
        np.ndarray
            Shape (N_samples, n_images, 128).
        """
        self._cnn_backbone.eval()
        all_encodings = []
        for seq in image_path_seq:
            enc_seq = []
            for path in seq:
                if path.exists():
                    img = np.load(path).astype(np.float32) / 255.0
                    img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        enc = self._cnn_backbone.conv_blocks(img_t).view(1, -1)
                    enc_seq.append(enc.cpu().numpy().flatten())
                else:
                    enc_seq.append(np.zeros(128, dtype=np.float32))
            all_encodings.append(np.stack(enc_seq))
        return np.array(all_encodings)

    def fit_from_path_seqs(
        self,
        path_seqs: list[list[Path]],
        targets: np.ndarray,
        regime_vecs: Optional[np.ndarray] = None,
    ) -> None:
        """Train CNN+LSTM from pre-built image path sequences."""
        encoded = self._encode_sequence(path_seqs)  # (N, n_images, 128)
        valid = ~np.isnan(targets)
        encoded_v = encoded[valid].astype(np.float32)
        targets_v = targets[valid].astype(np.float32)
        regime_v = regime_vecs[valid].astype(np.float32) if regime_vecs is not None else None

        n_val = max(1, int(0.1 * len(encoded_v)))
        X_tr, X_val = encoded_v[:-n_val], encoded_v[-n_val:]
        y_tr, y_val = targets_v[:-n_val], targets_v[-n_val:]
        r_tr = regime_v[:-n_val] if regime_v is not None else None
        r_val = regime_v[-n_val:] if regime_v is not None else None

        all_params = (
            list(self._cnn_backbone.parameters())
            + list(self._lstm_head.parameters())
            + list(self._output_fc.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=self.lr)
        criterion = nn.MSELoss()

        best_val = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._cnn_backbone.train()
            self._lstm_head.train()
            self._output_fc.train()

            perm = np.random.permutation(len(X_tr))
            for start in range(0, len(X_tr), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.from_numpy(X_tr[idx]).to(self.device)
                yb = torch.from_numpy(y_tr[idx]).to(self.device)
                optimizer.zero_grad()
                _, (h, _) = self._lstm_head(xb)
                h_last = self._dropout(h[-1])
                if r_tr is not None:
                    rb = torch.from_numpy(r_tr[idx]).to(self.device)
                    h_last = torch.cat([h_last, rb], dim=1)
                pred = self._output_fc(h_last).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

            # Validation
            self._cnn_backbone.eval()
            self._lstm_head.eval()
            self._output_fc.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(self.device)
                yv = torch.from_numpy(y_val).to(self.device)
                _, (hv, _) = self._lstm_head(xv)
                hv_last = hv[-1]
                if r_val is not None:
                    rv = torch.from_numpy(r_val).to(self.device)
                    hv_last = torch.cat([hv_last, rv], dim=1)
                val_loss = criterion(self._output_fc(hv_last).squeeze(-1), yv).item()

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug("CNN+LSTM early stop at epoch %d", epoch)
                    break

        self._fitted = True
        logger.info("CNNLSTMModel fitted: %d samples, best_val=%.6f", len(X_tr), best_val)

    def get_params(self) -> dict:
        return {
            "model": "CNNLSTMModel",
            "n_images": self.n_images,
            "step_days": self.step_days,
            "lstm_hidden": self.lstm_hidden,
            "regime_dim": self.regime_dim,
        }
