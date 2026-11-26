"""Turn per-minute anomaly scores into maintenance alarms and evaluate them per event.

Maintenance teams don't care about per-row accuracy. They care about:
  * Did we catch the failure? (event recall)
  * How much warning did we get? (lead time, measured from the first alarm in the
    look-back window to the reported failure start)
  * How often do we cry wolf? (false alarms per week on healthy operation)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import FAILURES, FailureEvent


def smooth(scores: pd.Series, window: str = "30min") -> pd.Series:
    return scores.rolling(window, min_periods=1).median()


def warmup_mask(has_data: pd.Series, minutes: int = 30) -> pd.Series:
    """True for the first `minutes` of every run of live data (cold starts after the
    train was parked / logger off). Scores there are unreliable and are ignored."""
    ok = has_data.to_numpy()
    run_id = pd.Series((ok[1:] != ok[:-1]).cumsum(), index=has_data.index[1:]).reindex(has_data.index, fill_value=0)
    pos_in_run = has_data.groupby(run_id.to_numpy()).cumcount()
    return has_data & (pos_in_run < minutes)


def alarms_from_scores(scores: pd.Series, threshold: float, persistence: str = "15min",
                       cooldown: str = "6h") -> pd.DatetimeIndex:
    """An alarm fires when the smoothed score stays above threshold for `persistence`.
    After firing, the same alarm is suppressed for `cooldown` to avoid alert storms."""
    above = (scores > threshold).astype(float)
    sustained = above.rolling(persistence, min_periods=1).min() >= 1.0
    fired, last = [], None
    cd = pd.Timedelta(cooldown)
    for ts in sustained.index[sustained.to_numpy()]:
        if last is None or ts - last >= cd:
            fired.append(ts)
            last = ts
    return pd.DatetimeIndex(fired)


@dataclass
class EventResult:
    failure_id: int
    detected: bool
    lead_time_h: float | None  # hours between first alarm and failure start (negative = after start)


def evaluate_alarms(alarms: pd.DatetimeIndex, index: pd.DatetimeIndex, lookback: str = "24h",
                    failures: list[FailureEvent] = FAILURES) -> dict:
    lb = pd.Timedelta(lookback)
    results, matched = [], pd.Series(False, index=alarms)
    for ev in failures:
        win = (alarms >= ev.start - lb) & (alarms <= ev.end)
        hit = alarms[win]
        matched[win] = True
        lead = (ev.start - hit.min()).total_seconds() / 3600 if len(hit) else None
        results.append(EventResult(ev.id, bool(len(hit)), lead))

    false_alarms = int((~matched).sum())
    # Weeks of healthy, observed operation (exclude failure + look-back windows).
    healthy = pd.Series(True, index=index)
    for ev in failures:
        healthy[(index >= ev.start - lb) & (index <= ev.end + pd.Timedelta("24h"))] = False
    step = index.to_series().diff().median()
    healthy_weeks = healthy.sum() * step / pd.Timedelta("7D")
    detected = [r for r in results if r.detected]
    fa_rate = false_alarms / healthy_weeks if healthy_weeks else float("nan")
    # Probability that an alarm process firing at random (Poisson, same rate) lands in a
    # look-back+failure window by luck. Recall well above this = real signal.
    import math
    win_h = [(lb + (ev.end - ev.start)).total_seconds() / 3600 for ev in failures]
    chance = float(sum(1 - math.exp(-fa_rate * h / 168) for h in win_h) / len(win_h))
    return {
        "chance_recall": chance,
        "events": results,
        "event_recall": len(detected) / len(failures),
        "median_lead_time_h": float(pd.Series([r.lead_time_h for r in detected]).median()) if detected else None,
        "false_alarms": false_alarms,
        "healthy_weeks": float(healthy_weeks),
        "false_alarms_per_week": fa_rate,
    }
