"""Extract two real 60-minute windows (healthy + during failure #4) for API tests and examples."""
import json, sys
import pandas as pd
sys.path.insert(0, "src")
from pdm.data import ANALOG
df = pd.read_parquet("data/interim/metropt3_1min.parquet")
def win(end):
    w = df.loc[:end, ANALOG].dropna().tail(60)
    assert len(w) == 60 and (w.index[-1] - w.index[0]) == pd.Timedelta("59min"), w.index[[0, -1]]
    return [{k: round(float(v), 4) for k, v in r.items()} for r in w.to_dict("records")]
out = {"healthy": win("2020-08-10 14:00"), "failure": win("2020-06-05 20:40")}
json.dump(out, open("tests/fixtures/windows.json", "w"))
import numpy as np
from pdm.serving import CompressorScorer
s = CompressorScorer()
for k, v in out.items():
    r = s.score(np.array([[x[c] for c in s.columns] for x in v]))
    print(k, round(r["score_ratio"], 2), r["anomalous"], max(r["sensor_contribution"], key=r["sensor_contribution"].get))
