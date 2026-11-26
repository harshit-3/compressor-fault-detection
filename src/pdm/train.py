"""End-to-end training + evaluation. Run from repo root:  python -m pdm.train   (PYTHONPATH=src)

Protocol (no leakage):
  * Unsupervised detectors are fit on healthy data only.
  * Alarm thresholds are calibrated on held-out healthy data (no failure labels).
  * v1 "static": fit once on Feb-Mar, frozen for the rest of the year.
  * v2 "rolling": every 14 days, refit on the trailing 60 days of data, excluding any
    failure that has *already been reported* at that point in time (what a real
    maintenance team would know). This adapts to seasonal drift (summer oil temps).
  * All four reported failures (Apr 18 - Jul 15) are unseen test events.
  * The supervised LightGBM model is evaluated with leave-one-failure-out CV.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd

from .alarms import alarms_from_scores, evaluate_alarms, smooth, warmup_mask
from .data import ANALOG, FAILURES, label_failures
from .features import build_features
from .models import DutyCycleRule, IForestDetector, LSTMAutoencoder, train_gbm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "metropt3_1min.parquet"
ART = ROOT / "artifacts"
FIG = ROOT / "reports" / "figures"

TRAIN_END = pd.Timestamp("2020-03-31 23:59")
CALIB_END = pd.Timestamp("2020-04-10 23:59")
CALIB_DAYS = 10
TRAIN_DAYS = 60
RETRAIN_EVERY = "14D"
CALIB_QUANTILE = 0.999
GBM_HORIZON = "6h"
POST_FAILURE_EXCLUDE = pd.Timedelta("3D")


def event_summary(ev: dict) -> dict:
    return {
        "event_recall": ev["event_recall"],
        "chance_recall": round(ev["chance_recall"], 3),
        "median_lead_time_h": None if ev["median_lead_time_h"] is None else round(ev["median_lead_time_h"], 1),
        "false_alarms_per_week": round(ev["false_alarms_per_week"], 3),
        "per_event": [{"id": r.failure_id, "detected": r.detected,
                       "lead_time_h": None if r.lead_time_h is None else round(r.lead_time_h, 1)}
                      for r in ev["events"]],
    }


def known_failure_mask(index: pd.DatetimeIndex, as_of: pd.Timestamp) -> np.ndarray:
    """Rows to exclude from training: failures already reported before `as_of`, with the
    48h before and POST_FAILURE_EXCLUDE after (repair / recommissioning)."""
    bad = np.zeros(len(index), dtype=bool)
    for ev in FAILURES:
        if ev.end < as_of:
            bad |= (index >= ev.start - pd.Timedelta("48h")) & (index <= ev.end + POST_FAILURE_EXCLUDE)
    return bad


def fit_and_score(factory, fit_rows, calib_rows, score_rows, warm):
    det = factory()
    det.fit(fit_rows)
    s = det.score(pd.concat([calib_rows, score_rows]) if calib_rows is not None else score_rows)
    s = s[~s.index.duplicated()]
    s[warm.reindex(s.index, fill_value=False).to_numpy()] = np.nan
    s = smooth(s.dropna())
    return det, s


def run_detector(name, factory, data, warm, test_index, mode: str):
    """mode='static' -> one fit; mode='rolling' -> refit every RETRAIN_EVERY."""
    t0 = time.time()
    pieces, thresholds, det = [], [], None
    if mode == "static":
        starts = [CALIB_END]
        ends = [data.index.max()]
    else:
        starts = list(pd.date_range(CALIB_END, data.index.max(), freq=RETRAIN_EVERY))
        ends = starts[1:] + [data.index.max()]
    for start, end in zip(starts, ends, strict=False):
        if mode == "static":
            fit_rows = data[data.index <= TRAIN_END]
            calib_rows = data[(data.index > TRAIN_END) & (data.index <= CALIB_END)]
        else:
            window = data[(data.index > start - pd.Timedelta(days=TRAIN_DAYS)) & (data.index <= start)]
            window = window[~known_failure_mask(window.index, start)]
            cut = start - pd.Timedelta(days=CALIB_DAYS)
            fit_rows, calib_rows = window[window.index <= cut], window[window.index > cut]
        score_rows = data[(data.index > start) & (data.index <= end)]
        det, s = fit_and_score(factory, fit_rows, calib_rows, score_rows, warm)
        thr = float(s[s.index <= start].quantile(CALIB_QUANTILE))
        thresholds.append(thr)
        part = s[s.index > start]
        pieces.append((part, thr))
    alarms = pd.DatetimeIndex(sorted(set().union(*[alarms_from_scores(p, t) for p, t in pieces])))
    ev = evaluate_alarms(alarms, test_index)
    score = pd.concat([p / t for p, t in pieces])  # normalised: 1.0 == threshold
    res = {"model": f"{name} ({mode})", "n_alarms": len(alarms), "n_refits": len(pieces),
           "seconds": round(time.time() - t0, 1), **event_summary(ev)}
    return res, score, det, thresholds[-1]


def run_gbm_lofo(X: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Leave-one-failure-out. Threshold per fold = 99.9th pct of the fold's *training*
    healthy predictions, so no information from the held-out event is used."""
    y = (labels["in_failure"] | labels["pre_failure"]).loc[X.index].astype(int)
    oof, per_event, fa, weeks = [], [], 0, 0.0
    for ev in FAILURES:
        test_mask = (X.index >= ev.start - pd.Timedelta("7D")) & (X.index <= ev.end + pd.Timedelta("2D"))
        buffer = (X.index >= ev.start - pd.Timedelta("8D")) & (X.index <= ev.end + pd.Timedelta("3D"))
        model = train_gbm(X[~buffer], y[~buffer])
        healthy_train = (~buffer) & (y.to_numpy() == 0)
        thr = float(np.quantile(model.predict_proba(X[healthy_train])[:, 1], CALIB_QUANTILE))
        p = smooth(pd.Series(model.predict_proba(X[test_mask])[:, 1], index=X.index[test_mask]))
        oof.append(p / thr)
        res = evaluate_alarms(alarms_from_scores(p, thr), p.index, failures=[ev])
        r = res["events"][0]
        per_event.append({"id": ev.id, "detected": r.detected,
                          "lead_time_h": None if r.lead_time_h is None else round(r.lead_time_h, 1)})
        fa += res["false_alarms"]
        weeks += res["healthy_weeks"]
    detected = [e for e in per_event if e["detected"]]
    return {
        "model": "lightgbm early-warning (LOFO-CV)",
        "event_recall": len(detected) / len(FAILURES),
        "median_lead_time_h": float(np.median([e["lead_time_h"] for e in detected])) if detected else None,
        "false_alarms_per_week": round(fa / weeks, 3) if weeks else None,
        "per_event": per_event,
    }, pd.concat(oof).sort_index()


def plot_scores(timelines: dict[str, pd.Series], path: Path):
    fig, axes = plt.subplots(len(timelines), 1, figsize=(14, 2.4 * len(timelines)), sharex=True)
    for ax, (name, s) in zip(np.atleast_1d(axes), timelines.items(), strict=False):
        s = s[s.index > CALIB_END].resample("10min").max()
        ax.plot(s.index, s.values, lw=0.6, color="#2b6cb0")
        ax.axhline(1.0, color="#c53030", ls="--", lw=1, label="alarm threshold")
        for ev in FAILURES:
            ax.axvspan(ev.start, ev.end, color="#e53e3e", alpha=0.25)
            ax.axvspan(ev.start - pd.Timedelta("24h"), ev.start, color="#ed8936", alpha=0.15)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(s.min(), 1e-3))
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Normalised anomaly score (1.0 = alarm threshold). Red = reported failure, "
                 "orange = 24h before failure", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ART.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("compressor-pdm")

    df = pd.read_parquet(DATA)
    warm = warmup_mask(df["has_data"])
    X = build_features(df)
    labels = label_failures(X.index, horizon=GBM_HORIZON)
    test_index = X.index[X.index > CALIB_END]
    print(f"features: {X.shape}, warm-up minutes masked: {int(warm.sum())}")

    factories = {
        "duty_cycle_rule": (DutyCycleRule, X),
        "isolation_forest": (IForestDetector, X),
        "lstm_autoencoder": (lambda: LSTMAutoencoder(ANALOG), df),
    }
    results, timelines = [], {}
    for name, (factory, data) in factories.items():
        for mode in ("static", "rolling"):
            with mlflow.start_run(run_name=f"{name}-{mode}"):
                res, score, det, thr = run_detector(name, factory, data, warm, test_index, mode)
                mlflow.log_params({"model": name, "mode": mode, "calib_quantile": CALIB_QUANTILE,
                                   "train_days": TRAIN_DAYS, "retrain_every": RETRAIN_EVERY})
                mlflow.log_metrics({k: v for k, v in res.items()
                                    if isinstance(v, (int, float)) and v is not None})
                print(json.dumps({k: v for k, v in res.items() if k != "per_event"}),
                      [(e["id"], e["lead_time_h"]) for e in res["per_event"]])
                results.append(res)
                timelines[res["model"]] = score
                if mode == "rolling" and name == "isolation_forest":
                    joblib.dump({"detector": det, "threshold": thr, "features": list(X.columns)},
                                ART / "isolation_forest.joblib")
                if mode == "static" and name == "lstm_autoencoder":
                    # Production model: weights (state_dict) + scaler + threshold, no pickled classes.
                    import torch
                    torch.save({"state_dict": {k: v.cpu() for k, v in det.net.state_dict().items()},
                                "columns": ANALOG, "seq_len": det.seq_len,
                                "scaler_mean": det.scaler.mean_.tolist(),
                                "scaler_scale": det.scaler.scale_.tolist(),
                                "threshold": thr, "trained_until": str(TRAIN_END)},
                               ART / "lstm_autoencoder.pt")

    # Ensemble: alarm only when BOTH detectors agree (min of normalised scores > 1).
    # Two detectors with different inductive biases rarely false-alarm at the same time.
    with mlflow.start_run(run_name="ensemble"):
        a = timelines["lstm_autoencoder (static)"]
        b = timelines["isolation_forest (rolling)"]
        both = pd.concat([a, b], axis=1, join="inner").min(axis=1)
        alarms = alarms_from_scores(both, 1.0)
        res = {"model": "ensemble: LSTM-AE AND IForest", "n_alarms": len(alarms),
               **event_summary(evaluate_alarms(alarms, test_index))}
        mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, (int, float)) and v is not None})
        print(json.dumps({k: v for k, v in res.items() if k != "per_event"}),
              [(e["id"], e["lead_time_h"]) for e in res["per_event"]])
        results.append(res)
        timelines[res["model"]] = both

    with mlflow.start_run(run_name="lightgbm-lofo"):
        res, oof = run_gbm_lofo(X, labels)
        mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, (int, float)) and v is not None})
        print(json.dumps({k: v for k, v in res.items() if k != "per_event"}),
              [(e["id"], e["lead_time_h"]) for e in res["per_event"]])
        results.append(res)
        timelines[res["model"]] = oof
        y = (labels["in_failure"] | labels["pre_failure"]).astype(int)
        final = train_gbm(X, y)
        imp = pd.Series(final.feature_importances_, index=X.columns).sort_values(ascending=False)
        imp.head(25).to_csv(ART / "gbm_feature_importance.csv")

    # Operating curve: sweep the alarm threshold (multiples of the calibrated one) to show
    # the detection vs false-alarm trade-off a maintenance planner would choose from.
    curve = []
    for name in ["lstm_autoencoder (static)", "isolation_forest (rolling)", "ensemble: LSTM-AE AND IForest"]:
        s = timelines[name]
        for m in [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0]:
            e = evaluate_alarms(alarms_from_scores(s, m), test_index)
            curve.append({"model": name, "multiplier": m, "event_recall": e["event_recall"],
                          "false_alarms_per_week": e["false_alarms_per_week"]})
    curve = pd.DataFrame(curve)
    curve.to_csv(ART / "operating_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, g in curve.groupby("model"):
        ax.plot(g.false_alarms_per_week, g.event_recall, marker="o", label=name)
    ax.set_xlabel("False alarms per week")
    ax.set_ylabel("Failures detected (of 4)")
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "1", "2", "3", "4"])
    ax.set_title("Operating curve: detection vs nuisance alarms")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "operating_curve.png", dpi=130)
    plt.close(fig)

    keep = ["duty_cycle_rule (rolling)", "isolation_forest (static)", "isolation_forest (rolling)",
            "lstm_autoencoder (static)", "ensemble: LSTM-AE AND IForest", "lightgbm early-warning (LOFO-CV)"]
    plot_scores({k: timelines[k] for k in keep if k in timelines}, FIG / "score_timelines.png")
    (ART / "metrics.json").write_text(json.dumps(results, indent=2))
    # Compact replay data for the dashboard: hourly sensor means + normalised production score.
    prod = timelines["lstm_autoencoder (static)"].rename("score")
    replay = df[ANALOG].resample("10min").mean().join(prod.resample("10min").max())
    replay = replay[replay.index > CALIB_END].dropna(how="all")
    replay.astype("float32").to_parquet(ROOT / "reports" / "replay.parquet")
    pd.set_option("display.width", 200)
    print("\nSUMMARY")
    print(pd.DataFrame(results)[["model", "event_recall", "chance_recall", "median_lead_time_h",
                                 "false_alarms_per_week"]].to_string(index=False))


if __name__ == "__main__":
    main()
