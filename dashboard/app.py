"""Maintenance-planner dashboard.  Run:  streamlit run dashboard/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm.alarms import alarms_from_scores, evaluate_alarms  # noqa: E402
from pdm.data import FAILURES  # noqa: E402
from pdm.serving import CompressorScorer  # noqa: E402

st.set_page_config(page_title="Compressor Fault Detection", layout="wide")


@st.cache_data
def load():
    replay = pd.read_parquet(ROOT / "reports" / "replay.parquet")
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    windows = json.loads((ROOT / "tests" / "fixtures" / "windows.json").read_text())
    return replay, metrics, windows


@st.cache_resource
def scorer():
    return CompressorScorer()


replay, metrics, windows = load()

st.title("Compressor fault detection: fleet health monitor")
st.caption("LSTM autoencoder trained on 2 months of healthy operation only. Data: MetroPT-3 air production "
           "unit (real industrial telemetry, Feb–Sep 2020), a stand-in for haul-truck / mining fleet compressors.")

tab1, tab2, tab3 = st.tabs(["Alarm replay", "Score a window", "Model comparison"])

with tab1:
    c1, c2 = st.columns([1, 3])
    with c1:
        mult = st.slider("Alarm threshold (× calibrated)", 0.5, 3.0, 1.25, 0.05,
                         help="Higher = fewer nuisance alarms, but risk missing failures.")
        persistence = st.select_slider("Persistence", ["5min", "15min", "30min", "60min"], "15min")
        sensor = st.selectbox("Overlay sensor", ["Oil_temperature", "Motor_current", "TP3", "DV_pressure", "TP2"])
    score = replay["score"].dropna()
    alarms = alarms_from_scores(score, mult, persistence=persistence)
    ev = evaluate_alarms(alarms, score.index)
    with c1:
        st.metric("Failures detected", f"{sum(e.detected for e in ev['events'])} / {len(FAILURES)}")
        st.metric("False alarms / week", f"{ev['false_alarms_per_week']:.2f}")
        st.metric("Total alarms", len(alarms))
        st.dataframe(pd.DataFrame([{"failure": e.failure_id, "detected": e.detected,
                                    "hours before reported start": None if e.lead_time_h is None
                                    else round(e.lead_time_h, 1)} for e in ev["events"]]),
                     hide_index=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=score.index, y=score / mult, name="anomaly score (1 = alarm)",
                                 line=dict(width=1, color="#2b6cb0")))
        fig.add_trace(go.Scatter(x=replay.index, y=replay[sensor], name=sensor, yaxis="y2",
                                 line=dict(width=0.8, color="#a0aec0"), opacity=0.6))
        fig.add_hline(y=1, line_dash="dash", line_color="#c53030")
        for f in FAILURES:
            fig.add_vrect(x0=f.start, x1=f.end, fillcolor="#e53e3e", opacity=0.25, line_width=0,
                          annotation_text=f"failure {f.id}", annotation_position="top left")
        fig.add_trace(go.Scatter(x=alarms, y=[1] * len(alarms), mode="markers", name="alarm",
                                 marker=dict(symbol="triangle-down", size=11, color="#c53030")))
        fig.update_layout(height=520, yaxis=dict(type="log", title="score / threshold"),
                          yaxis2=dict(overlaying="y", side="right", title=sensor, showgrid=False,
                                      type="linear", nticks=6, tickformat=".0f"),
                          legend=dict(orientation="h", y=1.08), margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Scores the last 60 minutes of readings, the same call the REST API (`POST /score`) makes.")
    choice = st.radio("Example window", ["healthy", "failure"], horizontal=True,
                      format_func=lambda k: {"healthy": "Normal operation (10 Aug)",
                                             "failure": "Air leak in progress (5 Jun)"}[k])
    s = scorer()
    df = pd.DataFrame(windows[choice])[s.columns]
    out = s.score(df.to_numpy(dtype=np.float32))
    c1, c2 = st.columns([1, 2])
    c1.metric("Score / threshold", f"{out['score_ratio']:.2f}",
              "ANOMALY" if out["anomalous"] else "normal", delta_color="inverse" if out["anomalous"] else "off")
    c1.bar_chart(pd.Series(out["sensor_contribution"], name="share of error"), horizontal=True)
    c2.line_chart(df[["TP2", "TP3", "Reservoirs", "Motor_current"]])

with tab3:
    st.write("All detectors are evaluated on the same four unseen air-leak failures. "
             "*Chance* = recall a random alarm generator with the same false-alarm rate would get.")
    tbl = pd.DataFrame(metrics)[["model", "event_recall", "chance_recall", "median_lead_time_h",
                                 "false_alarms_per_week"]]
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.image(str(ROOT / "reports" / "figures" / "operating_curve.png"))
