import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdm.api import app

WINDOWS = json.loads((Path(__file__).parent / "fixtures" / "windows.json").read_text())
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info():
    info = client.get("/model").json()
    assert info["window_minutes"] == 60
    assert len(info["sensors"]) == 7


def test_healthy_window_not_anomalous():
    r = client.post("/score", json={"readings": WINDOWS["healthy"]})
    assert r.status_code == 200
    body = r.json()
    assert body["anomalous"] is False
    assert body["score_ratio"] < 1
    assert abs(sum(body["sensor_contribution"].values()) - 1) < 1e-3


def test_failure_window_anomalous():
    r = client.post("/score", json={"asset_id": "HT-042", "readings": WINDOWS["failure"]})
    body = r.json()
    assert body["anomalous"] is True
    assert body["score_ratio"] > 2
    assert body["asset_id"] == "HT-042"


@pytest.mark.parametrize("n", [0, 59, 61])
def test_wrong_window_length_rejected(n):
    readings = (WINDOWS["healthy"] * 2)[:n]
    assert client.post("/score", json={"readings": readings}).status_code == 422


def test_missing_sensor_rejected():
    bad = [dict(r) for r in WINDOWS["healthy"]]
    del bad[0]["TP2"]
    assert client.post("/score", json={"readings": bad}).status_code == 422
