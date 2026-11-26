"""Feature engineering on the resampled (1-min) compressor signals.

Two families of features:
1. Rolling statistics of analogue sensors (level, spread, trend) over several windows.
2. Physics-informed features. An air leak means the compressor has to work harder to
   hold reservoir pressure, so we track duty cycle, load/unload switching rate,
   pressure differentials and how fast pressure decays while the motor is idle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ANALOG

WINDOWS = ["15min", "60min", "240min"]


def physics_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    # Compressor "under load" when motor current is high (~7A per spec).
    loaded = (df["Motor_current"] > 5.5).astype("float32")
    f["loaded"] = loaded
    f["tp3_minus_reservoir"] = df["TP3"] - df["Reservoirs"]
    f["tp2_minus_tp3"] = df["TP2"] - df["TP3"]
    # Pressure change per minute: negative while idle = air being consumed or leaking.
    f["tp3_slope"] = df["TP3"].diff()
    f["idle_tp3_slope"] = f["tp3_slope"].where(loaded == 0)
    return f


def rolling_features(df: pd.DataFrame, windows: list[str] = WINDOWS) -> pd.DataFrame:
    base = pd.concat([df[ANALOG], physics_features(df)], axis=1)
    feats = {}
    for w in windows:
        r = base.rolling(w, min_periods=5)
        mean, std = r.mean(), r.std()
        for c in base.columns:
            feats[f"{c}__mean_{w}"] = mean[c]
            feats[f"{c}__std_{w}"] = std[c]
        # Duty cycle and load/unload switching rate are the strongest leak indicators.
        feats[f"duty_cycle_{w}"] = base["loaded"].rolling(w, min_periods=5).mean()
        feats[f"switches_{w}"] = base["loaded"].diff().abs().rolling(w, min_periods=5).sum()
    out = pd.DataFrame(feats, index=df.index)
    # Trend: short-window mean minus long-window mean for key signals.
    for c in ["Oil_temperature", "Motor_current", "TP3", "idle_tp3_slope"]:
        out[f"{c}__trend"] = out[f"{c}__mean_{windows[0]}"] - out[f"{c}__mean_{windows[-1]}"]
    out["duty_trend"] = out[f"duty_cycle_{windows[0]}"] - out[f"duty_cycle_{windows[-1]}"]
    return out.astype("float32")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix; rows without live sensor data are dropped."""
    feats = rolling_features(df)
    feats = feats[df["has_data"]]
    return feats.replace([np.inf, -np.inf], np.nan).dropna()
