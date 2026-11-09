"""Out-of-regime test: score the FINAL models (trained on 33 days of the 43-day table, results_table_43days/) on the 7 active event days
(22-23 Feb 2024, 7-11 May 2024) that they never saw. Reads dataset/training_table.csv. No training happens here.
Writes results_event_days/results_event_days.json
"""
import json, os, warnings
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); SEED = 20260718
MODELS = os.path.join(HERE, "results_table_43days"); OUT = os.path.join(HERE, "results_event_days"); os.makedirs(OUT, exist_ok=True)
t = pd.read_csv(os.path.join(HERE, "dataset", "training_table.csv"), index_col=0, parse_dates=True, low_memory=False)
valid = t.has_data.to_numpy(bool); dates = t.date.to_numpy()
ev_mask = (t.period.to_numpy() == "events_2024") & valid
tr43 = (t.split.to_numpy() == "train") & (t.period.to_numpy() != "events_2024") & valid        # the rows the final models trained on

y_now = t.y_now.to_numpy(bool); gap = np.r_[True, np.diff(t.index.values).astype("timedelta64[s]").astype(int) > 60]
start = y_now & (~np.r_[False, y_now[:-1]] | gap); run = np.cumsum(start) - 1
ev_id = np.where(y_now, run, -1); starts = np.where(start)[0]; nxt = np.searchsorted(starts, np.arange(len(t)), side="right")
on_id = np.where(t.y_forecast_30min.to_numpy(bool) & (nxt < len(starts)), nxt, -1)

def runs_of(b):
    r, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j + 1 < n and b[j + 1]: j += 1
            r.append((i, j)); i = j + 1
        else: i += 1
    return r
def tss(y, yh): tp = (y & yh).sum(); fn = (y & ~yh).sum(); fp = (~y & yh).sum(); tn = (~y & ~yh).sum(); return tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
def rank01(p, ref): return np.searchsorted(np.sort(ref), p) / len(ref)

results = {}
for task, ycol, idarr in (("NOWCAST", "y_now", ev_id), ("FORECAST", "y_forecast_30min", on_id)):
    bundle = joblib.load(os.path.join(MODELS, f"models_{task.lower()}.joblib")); y = t[ycol].to_numpy(bool); yte = y[ev_mask]
    scores = {}
    for n, m in bundle["members"].items():
        X = t[m["features"]].fillna(0).to_numpy(float)[ev_mask]; scores[n] = (m["model"].predict_proba(X)[:, 1], m["threshold"])
    scores["ENSEMBLE"] = (np.mean([rank01(scores[n][0], bundle["members"][n]["oof_ref"]) for n in bundle["members"]], axis=0), bundle["ensemble_threshold"])
    lr_tr = t.log_rate.to_numpy()[tr43]; ytr = y[tr43]; q = np.quantile(lr_tr, np.linspace(0.5, 0.999, 200))
    scores["threshold_rate"] = (t.log_rate.to_numpy()[ev_mask], float(q[int(np.argmax([f1_score(ytr, lr_tr >= v, zero_division=0) for v in q]))]))
    scores["persistence"] = (t.groupby("date").y_now.shift(1).fillna(0).to_numpy(float)[ev_mask], 0.5)
    ids, dte = idarr[ev_mask], dates[ev_mask]; res = {}
    print(f"\n=== {task} on the 7 event days: {ev_mask.sum()} minutes, positives {yte.sum()} ({yte.mean():.1%}), flares {len(set(ids[ids >= 0]))} ===")
    for n, (p, thr) in scores.items():
        yh = p >= thr; ev = set(ids[ids >= 0]); det = set(ids[(ids >= 0) & yh]); rr = runs_of(yh)
        fr = sum(1 for a, b in rr if not (ids[a:b + 1] >= 0).any()); rng = np.random.default_rng(SEED); ud = np.unique(dte); au = []
        for _ in range(1000):
            ix = np.concatenate([np.where(dte == d)[0] for d in rng.choice(ud, len(ud))])
            if 0 < yte[ix].sum() < len(ix): au.append(roc_auc_score(yte[ix], p[ix]))
        res[n] = {"roc_auc": float(roc_auc_score(yte, p)), "roc_auc_ci95": [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
                  "pr_auc": float(average_precision_score(yte, p)), "tss": float(tss(yte, yh)), "n_events": len(ev), "events_detected": len(det),
                  "alert_runs": len(rr), "false_alert_runs": fr, "threshold": float(thr)}
        r = res[n]; print(f"  {n:18s} AUC={r['roc_auc']:.3f} [{r['roc_auc_ci95'][0]:.2f},{r['roc_auc_ci95'][1]:.2f}] PR={r['pr_auc']:.3f} TSS={r['tss']:.2f} events={r['events_detected']}/{r['n_events']} false_runs={fr}/{len(rr)}")
    results[task] = {"base_rate": float(yte.mean()), "models": res}
json.dump({"test": "events_2024 (7 days)", "models_from": "results_table_43days", "results": results}, open(os.path.join(OUT, "results_event_days.json"), "w"), indent=1)
print("\nwritten", OUT)
