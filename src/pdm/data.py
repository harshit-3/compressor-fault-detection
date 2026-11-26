"""Data loading, resampling and failure labelling for the MetroPT-3 compressor dataset.

The raw CSV is ~1.5M rows sampled every ~10 s. We convert it once to Parquet and
resample to a fixed grid so every downstream step works on a regular time series.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
SENSORS = ANALOG + DIGITAL


@dataclass(frozen=True)
class FailureEvent:
    id: int
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str = "Air leak"


# From the dataset's official failure report (Data Description_Metro.pdf).
FAILURES: list[FailureEvent] = [
    FailureEvent(1, pd.Timestamp("2020-04-18 00:00"), pd.Timestamp("2020-04-18 23:59")),
    FailureEvent(2, pd.Timestamp("2020-05-29 23:30"), pd.Timestamp("2020-05-30 06:00")),
    FailureEvent(3, pd.Timestamp("2020-06-05 10:00"), pd.Timestamp("2020-06-07 14:30")),
    FailureEvent(4, pd.Timestamp("2020-07-15 14:30"), pd.Timestamp("2020-07-15 19:00")),
]


def csv_to_parquet(csv_path: Path, parquet_path: Path) -> Path:
    df = pd.read_csv(csv_path, usecols=["timestamp", *SENSORS], parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df[SENSORS] = df[SENSORS].astype("float32")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def load_raw(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path).set_index("timestamp")


def resample(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Mean-resample to a regular grid. Gaps (train parked / logger off) stay NaN and are flagged."""
    out = df[SENSORS].resample(rule).mean()
    out["has_data"] = out[ANALOG].notna().all(axis=1)
    return out


def label_failures(index: pd.DatetimeIndex, horizon: str = "2h") -> pd.DataFrame:
    """Per-timestamp labels.

    - in_failure:     timestamp is inside a reported failure window
    - pre_failure:    timestamp is within `horizon` before a failure starts (early-warning target)
    - failure_id:     which event (0 = none), used for leave-one-failure-out CV
    """
    h = pd.Timedelta(horizon)
    labels = pd.DataFrame(index=index)
    labels["in_failure"] = False
    labels["pre_failure"] = False
    labels["failure_id"] = 0
    for ev in FAILURES:
        in_win = (index >= ev.start) & (index <= ev.end)
        pre_win = (index >= ev.start - h) & (index < ev.start)
        labels.loc[in_win, "in_failure"] = True
        labels.loc[pre_win, "pre_failure"] = True
        labels.loc[in_win | pre_win, "failure_id"] = ev.id
    return labels
