"""Load the production LSTM autoencoder and score sensor windows (CPU, no training deps)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models import _LSTMAE

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "artifacts" / "lstm_autoencoder.pt"


class CompressorScorer:
    def __init__(self, path: Path = DEFAULT_MODEL):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.columns: list[str] = ckpt["columns"]
        self.seq_len: int = ckpt["seq_len"]
        self.mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
        self.scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        self.threshold: float = ckpt["threshold"]
        self.trained_until: str = ckpt["trained_until"]
        self.net = _LSTMAE(len(self.columns))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

    @torch.no_grad()
    def score(self, window: np.ndarray) -> dict:
        """window: [seq_len, n_sensors] of 1-minute means in `self.columns` order.
        Returns the reconstruction error, its ratio to the alarm threshold, and the
        per-sensor share of the error (which sensor looks most abnormal)."""
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (self.seq_len, len(self.columns)):
            raise ValueError(f"expected shape ({self.seq_len}, {len(self.columns)}), got {window.shape}")
        if not np.isfinite(window).all():
            raise ValueError("window contains NaN/inf")
        x = torch.from_numpy((window - self.mean) / self.scale).unsqueeze(0)
        err = ((self.net(x) - x) ** 2).squeeze(0).mean(dim=0).numpy()  # per-sensor MSE
        score = float(err.mean())
        contrib = err / err.sum() if err.sum() > 0 else err
        return {
            "score": score,
            "threshold": self.threshold,
            "score_ratio": score / self.threshold,
            "anomalous": score > self.threshold,
            "sensor_contribution": {c: round(float(v), 4) for c, v in zip(self.columns, contrib, strict=False)},
        }
