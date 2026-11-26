import numpy as np
import pandas as pd

from pdm.alarms import alarms_from_scores, evaluate_alarms, warmup_mask
from pdm.data import FailureEvent, label_failures
from pdm.features import build_features


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="1min"), dtype=float)


def test_alarm_requires_persistence():
    s = _series([0] * 10 + [5] * 5 + [0] * 10)  # 5-minute blip
    assert len(alarms_from_scores(s, 1.0, persistence="15min")) == 0
    s = _series([0] * 10 + [5] * 20 + [0] * 10)  # 20-minute excursion
    assert len(alarms_from_scores(s, 1.0, persistence="15min")) == 1


def test_alarm_cooldown_suppresses_repeats():
    s = _series([5] * 600)  # 10 hours continuously high
    alarms = alarms_from_scores(s, 1.0, persistence="15min", cooldown="6h")
    assert len(alarms) == 2


def test_evaluate_lead_time_and_false_alarms():
    idx = pd.date_range("2020-01-01", "2020-03-01", freq="1min")
    ev = FailureEvent(1, pd.Timestamp("2020-02-01 12:00"), pd.Timestamp("2020-02-01 18:00"))
    alarms = pd.DatetimeIndex(["2020-02-01 09:00", "2020-01-10 00:00"])
    res = evaluate_alarms(alarms, idx, lookback="24h", failures=[ev])
    assert res["event_recall"] == 1.0
    assert res["events"][0].lead_time_h == 3.0
    assert res["false_alarms"] == 1
    assert 0 < res["chance_recall"] < 1


def test_warmup_mask_marks_start_of_each_run():
    has = _series([0] * 5 + [1] * 50 + [0] * 5 + [1] * 50).astype(bool)
    m = warmup_mask(has, minutes=30)
    assert m.sum() == 60
    assert not m.iloc[:5].any()
    assert m.iloc[5:35].all() and not m.iloc[35:55].any()


def test_labels_and_features_shape():
    idx = pd.date_range("2020-04-17", periods=3000, freq="1min")
    lab = label_failures(idx, horizon="2h")
    assert lab["pre_failure"].sum() == 120
    rng = np.random.default_rng(0)
    cols = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
    df = pd.DataFrame(rng.normal(5, 1, (3000, 7)), index=idx, columns=cols)
    df["has_data"] = True
    X = build_features(df)
    assert len(X) > 2500
    assert not X.isna().any().any()
