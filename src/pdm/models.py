"""Anomaly detectors and the supervised early-warning model.

* DutyCycleRule      - domain baseline: alarm when the compressor runs too much.
* IForestDetector    - Isolation Forest on engineered features (unsupervised).
* LSTMAutoencoder    - sequence autoencoder on raw 1-min signals, trained on healthy
                       data only; reconstruction error = anomaly score (GPU).
* train_gbm          - LightGBM early-warning classifier ("failure within horizon").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn


class DutyCycleRule:
    name = "duty_cycle_rule"

    def __init__(self, col: str = "duty_cycle_60min"):
        self.col = col

    def fit(self, X: pd.DataFrame):
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        return X[self.col]


class IForestDetector:
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, max_samples=4096,
                                     random_state=random_state, n_jobs=-1)
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame):
        self.columns = list(X.columns)
        self.model.fit(self.scaler.fit_transform(X.to_numpy()))
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        Z = self.scaler.transform(X[self.columns].to_numpy())
        return pd.Series(-self.model.score_samples(Z), index=X.index)


# ----------------------------------------------------------------------------- LSTM-AE
class _LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_latent(h[-1])
        rep = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(rep)
        return self.out(y)


def make_windows(arr: np.ndarray, seq_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Return windows [N, seq_len, F] and the index position of each window's last step."""
    ends = np.arange(seq_len - 1, len(arr), stride)
    idx = ends[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
    return arr[idx], ends


class LSTMAutoencoder:
    name = "lstm_autoencoder"

    def __init__(self, columns: list[str], seq_len: int = 60, epochs: int = 15,
                 batch_size: int = 256, lr: float = 1e-3, device: str | None = None, seed: int = 42):
        self.columns, self.seq_len, self.epochs = columns, seq_len, epochs
        self.batch_size, self.lr = batch_size, lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.seed = seed
        self.net: _LSTMAE | None = None
        self.history: list[float] = []

    def _segments(self, df: pd.DataFrame):
        """Split into contiguous runs of live data so windows never span logging gaps."""
        ok = df["has_data"].to_numpy()
        run_id = np.cumsum(np.r_[True, ok[1:] != ok[:-1]])
        for _, seg in df[ok].groupby(run_id[ok]):
            if len(seg) >= self.seq_len:
                yield seg

    def fit(self, df: pd.DataFrame):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        live = df[df["has_data"]]
        self.scaler.fit(live[self.columns].to_numpy())
        wins = [make_windows(self.scaler.transform(s[self.columns].to_numpy()), self.seq_len, 5)[0]
                for s in self._segments(df)]
        X = torch.tensor(np.concatenate(wins), dtype=torch.float32)
        self.net = _LSTMAE(len(self.columns)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            total = 0.0
            self.net.train()
            for i in range(0, len(X), self.batch_size):
                xb = X[perm[i:i + self.batch_size]].to(self.device)
                loss = loss_fn(self.net(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            self.history.append(total / len(X))
        return self

    @torch.no_grad()
    def score(self, df: pd.DataFrame) -> pd.Series:
        self.net.eval()
        parts = []
        for seg in self._segments(df):
            w, ends = make_windows(self.scaler.transform(seg[self.columns].to_numpy()), self.seq_len, 1)
            errs = []
            for i in range(0, len(w), 4096):
                xb = torch.tensor(w[i:i + 4096], dtype=torch.float32, device=self.device)
                errs.append(((self.net(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
            parts.append(pd.Series(np.concatenate(errs), index=seg.index[ends]))
        return pd.concat(parts).sort_index()


# ----------------------------------------------------------------------------- LightGBM
def train_gbm(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    import lightgbm as lgb

    pos = max(int(y.sum()), 1)
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        scale_pos_weight=(len(y) - pos) / pos, random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    return model
