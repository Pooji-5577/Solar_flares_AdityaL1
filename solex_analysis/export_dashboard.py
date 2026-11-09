"""Export SoLEXS results to a compact JSON consumed by the AdityaNet web dashboard (web/src/generated/data/solex/solex.json)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from solex_common import load_minutes, label_events
HERE = os.path.dirname(__file__); R = os.path.join(HERE, "results")
DEST = os.path.join(HERE, "..", "data_analysis", "web", "src", "generated", "data", "solex", "solex.json")
TEST = {"2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31"}; BIN = 5

df = load_minutes(); flare, events, sigma = label_events(df, k=4.0); df["flare"] = flare; df["date"] = df.index.strftime("%Y-%m-%d")
def binned(s, how): return s.groupby(np.arange(len(s)) // BIN).agg(how)
lc = {"bin_minutes": BIN, "log10_rate": [round(float(v), 3) for v in np.log10(binned(df.rate, "mean"))],
      "flare": [int(v) for v in binned(df.flare.astype(int), "max")], "start": df.index[0].isoformat(),
      "day_starts": [int(np.where(df.date.to_numpy() == d)[0][0] // BIN) for d in sorted(df.date.unique())], "days": sorted(df.date.unique())}
ev = []
for a, b in events:
    seg = df.rate.iloc[a:b + 1]; pk = seg.idxmax()
    ev.append({"start": df.index[a].strftime("%Y-%m-%d %H:%M"), "peak": pk.strftime("%H:%M"), "end": df.index[b].strftime("%H:%M"),
               "minutes": int(b - a + 1), "peak_rate": round(float(seg.max()), 1), "split": "test" if df.date.iloc[a] in TEST else "train"})
daily = [{"date": d, "median": round(float(g.rate.median()), 1), "max": round(float(g.rate.max()), 1), "flares": sum(1 for e in ev if e["start"].startswith(d))}
         for d, g in df.groupby("date")]
sc = pd.read_csv(f"{R}/test_scores_strong.csv", index_col=0, parse_dates=True); thr = json.load(open(f"{R}/thresholds_strong.json"))["thresholds"]
te = df.loc[sc.index]
def b5(x, how="max"): return [round(float(v), 3) for v in x.groupby(np.arange(len(x)) // BIN).agg(how)]
strip = {"start": sc.index[0].isoformat(), "bin_minutes": BIN, "log10_rate": [round(float(v), 3) for v in np.log10(te.rate.groupby(np.arange(len(te)) // BIN).mean())],
         "nowcast_score": b5(sc["NOWCAST_ENSEMBLE"]), "forecast_score": b5(sc["FORECAST_ENSEMBLE"]),
         "flare": [int(v) for v in b5(sc["y_now"].astype(int))], "pre_onset": [int(v) for v in b5(sc["y_pred"].astype(int))],
         "nowcast_threshold": thr["NOWCAST"]["ENSEMBLE"], "forecast_threshold": thr["FORECAST"]["ENSEMBLE"],
         "day_starts": [int(np.where(te.date.to_numpy() == d)[0][0] // BIN) for d in sorted(te.date.unique())], "days": sorted(te.date.unique())}
rs = json.load(open(f"{R}/results_strong.json")); first = json.load(open(f"{R}/results.json"))
out = {"meta": {"instrument": "Aditya-L1 SoLEXS (SDD2)", "days": int(df.date.nunique()), "range": [df.date.iloc[0], df.date.iloc[-1]],
                "minutes": int(len(df)), "events": len(events), "label_sigma": round(sigma, 3), "train_days": 6, "test_days": 5,
                "test_events": sum(e["split"] == "test" for e in ev), "horizon_min": 30, "features": rs["n_features"]},
       "lightcurve": lc, "events": ev, "daily": daily, "test_strip": strip, "results": rs["results"],
       "first_run": {k.split()[0]: {"models": {m: {x: v[x] for x in ("roc_auc", "tss")} for m, v in r["models"].items() if not m.startswith("_")}} for k, r in first["results"].items()}}
os.makedirs(os.path.dirname(DEST), exist_ok=True); json.dump(out, open(DEST, "w"), separators=(",", ":")); print("wrote", os.path.abspath(DEST), os.path.getsize(DEST) // 1024, "KB")
