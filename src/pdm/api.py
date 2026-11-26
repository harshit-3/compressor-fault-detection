"""FastAPI inference service.  Run:  uvicorn pdm.api:app --port 8000   (PYTHONPATH=src)

POST /score with the last 60 one-minute sensor readings -> anomaly score + per-sensor attribution.
Note: a single window is scored here; the alarm logic (30-min smoothing, 15-min persistence,
6h cooldown) runs on the stream of scores in the monitoring layer (see pdm.alarms).
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .serving import CompressorScorer

app = FastAPI(title="Compressor Fault Detection API", version="1.0.0",
              description="LSTM-autoencoder anomaly scoring for compressor / air-production units.")


class Reading(BaseModel):
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float


class ScoreRequest(BaseModel):
    asset_id: str = Field("APU-01", description="Equipment identifier")
    readings: list[Reading] = Field(..., description="Exactly 60 consecutive 1-minute mean readings, oldest first")


class ScoreResponse(BaseModel):
    asset_id: str
    score: float
    threshold: float
    score_ratio: float
    anomalous: bool
    top_sensor: str
    sensor_contribution: dict[str, float]
    latency_ms: float


@lru_cache
def get_scorer() -> CompressorScorer:
    return CompressorScorer()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    s = get_scorer()
    return {"model": "lstm_autoencoder", "sensors": s.columns, "window_minutes": s.seq_len,
            "threshold": s.threshold, "trained_until": s.trained_until}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    s = get_scorer()
    if len(req.readings) != s.seq_len:
        raise HTTPException(422, f"need exactly {s.seq_len} readings, got {len(req.readings)}")
    t0 = time.perf_counter()
    window = np.array([[getattr(r, c) for c in s.columns] for r in req.readings], dtype=np.float32)
    try:
        out = s.score(window)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    top = max(out["sensor_contribution"], key=out["sensor_contribution"].get)
    return ScoreResponse(asset_id=req.asset_id, top_sensor=top,
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2), **out)
