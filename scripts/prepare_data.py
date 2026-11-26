"""Download MetroPT-3 from the UCI repository and build the 1-minute dataset.

    python scripts/prepare_data.py
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdm.data import csv_to_parquet, load_raw, resample  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
RAW = ROOT / "data" / "raw" / "metropt3.csv"
PQ = ROOT / "data" / "interim" / "metropt3.parquet"
OUT = ROOT / "data" / "interim" / "metropt3_1min.parquet"

if not RAW.exists():
    print("downloading MetroPT-3 (~210 MB) ...")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    blob = requests.get(URL, timeout=600).content
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        RAW.write_bytes(z.read(name))
if not PQ.exists():
    csv_to_parquet(RAW, PQ)
df = resample(load_raw(PQ))
df.to_parquet(OUT)
print(f"wrote {OUT} rows={len(df)} live={df.has_data.mean():.1%}")
