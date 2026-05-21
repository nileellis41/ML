"""2D CNN candlestick image model.

Reads rendered candlestick chart images (64x60 pixels, 1 or 2 channels).
Learns visual price patterns analogous to human technical analysis but
without hand-coded indicators (Jiang/Kelly/Xiu 2023 approach).

Variants
--------
CNN_base   : image only
CNN_regime : image + 4-dim regime probability vector appended before FC head
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.base import ModelInterface

logger = logging.getLogger(__name__)

_IMG_H = 64
_IMG_W = 60


class _CandleDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        targets: np.ndarray,
        regime_vecs: Optional[np.ndarray] = None,
    ):
        self.paths = image_paths
        self.targets = targets.astype(np.float32)
        self.regime_vecs = regime_vecs.astype(np.float32) if regime_vecs is not None else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx]).astype(np.float32) / 255.0
        target = self.targets[idx]
        if self.regime_vecs is not None:
            return torch.from_numpy(img), torch.tensor(self.regime_vecs[idx]), torch.tensor(target)
        return torch.from_numpy(img), torch.tensor(target)


class _CNNNet(nn.Module):
    def __init__(self, in_channels: int = 2, dropout: float = 0.5, regime_dim: int = 0):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=(5, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(dropout)
        fc_in = 128 + regime_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.conv_blocks(x).view(x.size(0), -1)  # (B, 128)
        feats = self.dropout(feats)
        if regime is not None:
            feats = torch.cat([feats, regime], dim=1)
        return self.fc(feats).squeeze(-1)


class CNNCandlestickModel(ModelInterface):
    """2D CNN over rendered candlestick images.

    Parameters
    ----------
    in_channels:
        1 (OHLC only) or 2 (OHLC + volume).
    dropout:
        Dropout before FC head.
    lr:
        Adam learning rate.
    batch_size:
        Mini-batch size.
    epochs:
        Max training epochs.
    patience:
        Early-stopping patience.
    regime_dim:
        If > 0, concatenate a regime probability vector of this size to
        the flattened conv features before the dense head (_regime variant).
    loss:
        'mse' for return regression; 'bce' for directional classification.
    device:
        Torch device string; auto-detected if None.
    candle_cache_dir:
        Directory where {ticker}/{date}.npy files are stored.
    """

    def __init__(
        self,
        in_channels: int = 2,
        dropout: float = 0.5,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        regime_dim: int = 0,
        loss: str = "mse",
        device: Optional[str] = None,
        candle_cache_dir: Path = Path("data/processed/candles"),
    ):
        self.in_channels = in_channels
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.regime_dim = regime_dim
        self.loss_type = loss
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.candle_cache_dir = Path(candle_cache_dir)
        self._net: Optional[_CNNNet] = None
        self._fitted = False

    def _get_image_path(self, ticker: str, date: pd.Timestamp) -> Path:
        return self.candle_cache_dir / ticker / f"{date.strftime('%Y%m%d')}.npy"

    def fit_from_paths(
        self,
        image_paths: list[Path],
        targets: np.ndarray,
        regime_vecs: Optional[np.ndarray] = None,
    ) -> None:
        """Train the CNN from a list of pre-rendered image paths.

        Parameters
        ----------
        image_paths:
            Ordered list of .npy file paths, one per training sample.
        targets:
            Forward log returns, same order as image_paths.
        regime_vecs:
            Optional (N, regime_dim) array of regime probabilities.
        """
        valid = [
            i for i, p in enumerate(image_paths)
            if p.exists() and not np.isnan(targets[i])
        ]
        if len(valid) == 0:
            logger.warning("CNNCandlestickModel: no valid training samples. Skipping fit.")
            return

        paths_v = [image_paths[i] for i in valid]
        targets_v = targets[valid]
        regime_v = regime_vecs[valid] if regime_vecs is not None else None

        n_val = max(1, int(0.1 * len(paths_v)))
        train_ds = _CandleDataset(paths_v[:-n_val], targets_v[:-n_val], regime_v[:-n_val] if regime_v is not None else None)
        val_ds = _CandleDataset(paths_v[-n_val:], targets_v[-n_val:], regime_v[-n_val:] if regime_v is not None else None)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self._net = _CNNNet(self.in_channels, self.dropout, self.regime_dim).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        if self.loss_type == "bce":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self._net.train()
            for batch in train_loader:
                optimizer.zero_grad()
                if len(batch) == 3:
                    imgs, regime, tgt = (b.to(self.device) for b in batch)
                    pred = self._net(imgs, regime)
                else:
                    imgs, tgt = (b.to(self.device) for b in batch)
                    pred = self._net(imgs)
                loss = criterion(pred, tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_preds, val_tgts = [], []
                for batch in DataLoader(val_ds, batch_size=64):
                    if len(batch) == 3:
                        imgs, regime, tgt = (b.to(self.device) for b in batch)
                        out = self._net(imgs, regime)
                    else:
                        imgs, tgt = (b.to(self.device) for b in batch)
                        out = self._net(imgs)
                    val_preds.append(out.cpu())
                    val_tgts.append(tgt.cpu())
                val_loss = criterion(
                    torch.cat(val_preds), torch.cat(val_tgts)
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug("CNN early stop at epoch %d (val_loss=%.6f)", epoch, best_val_loss)
                    break

        if hasattr(self, "_best_state"):
            self._net.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})
        self._fitted = True
        logger.info("CNNCandlestickModel fitted: %d samples, best_val_loss=%.6f", len(paths_v), best_val_loss)

    # ModelInterface stubs -- CNN needs image paths, not raw arrays.
    # Use fit_from_paths() directly for training.
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError(
            "CNNCandlestickModel.fit() expects image paths. Use fit_from_paths() instead."
        )

    def predict_from_paths(
        self,
        image_paths: list[Path],
        regime_vecs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return predictions for a list of image paths."""
        if not self._fitted or self._net is None:
            raise RuntimeError("CNNCandlestickModel has not been fitted.")

        valid_mask = np.array([p.exists() for p in image_paths])
        preds = np.full(len(image_paths), np.nan)

        if valid_mask.sum() == 0:
            return preds

        paths_v = [image_paths[i] for i in range(len(image_paths)) if valid_mask[i]]
        regime_v = regime_vecs[valid_mask] if regime_vecs is not None else None
        dummy_targets = np.zeros(len(paths_v))
        ds = _CandleDataset(paths_v, dummy_targets, regime_v)

        self._net.eval()
        all_preds = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=64):
                if len(batch) == 3:
                    imgs, regime, _ = (b.to(self.device) for b in batch)
                    out = self._net(imgs, regime)
                else:
                    imgs, _ = (b.to(self.device) for b in batch)
                    out = self._net(imgs)
                all_preds.append(out.cpu().numpy())

        preds[valid_mask] = np.concatenate(all_preds)
        return preds

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "CNNCandlestickModel.predict() expects image paths. Use predict_from_paths() instead."
        )

    def get_params(self) -> dict:
        return {
            "model": "CNNCandlestickModel",
            "in_channels": self.in_channels,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "regime_dim": self.regime_dim,
            "loss": self.loss_type,
        }
