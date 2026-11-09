"""SoLEXS-only nowcasting + forecasting, following AdityaNet's benchmark protocol (research/scripts/v2/ml/benchmark.py).

Labels are self-derived from SoLEXS (non-causal detector, solex_common.label_events); features are causal (trailing
windows only). Split is chronological by day. Thresholds are chosen on validation only; test is opened once.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
sys.path.insert(0, os.path.dirname(__file__))
from solex_common import load_minutes, label_events
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(OUT, exist_ok=True)
SEED, HORIZON = 20260718, 30
TRAIN = ["2025-03-20", "2025-03-22", "2025-03-23", "2025-03-24", "2025-03-25"]
VAL = ["2025-03-26"]
TEST = ["2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31"]
FEATURES = ["log_rate", "roll_mean_5", "roll_mean_15", "roll_mean_30", "roll_mean_60", "roll_max_15",
            "roll_std_15", "roll_std_60", "bg_excess", "rise_5", "rise_15"]

df = load_minutes()
df["date"] = df.index.strftime("%Y-%m-%d")
flare, events, sigma = label_events(df, k=4.0)
df["y_now"] = flare
starts = np.array([a for a, _ in events]); ends = np.array([b for _, b in events])
# forecast target: a flare ONSET falls in (t, t+30 min]
idx = np.arange(len(df)); nxt = np.searchsorted(starts, idx, side="right")
nxt_start = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], 10**9)
same_day = np.array([df.date.iloc[min(s, len(df)-1)] == d for s, d in zip(nxt_start, df.date)])
df["y_pred"] = (nxt_start - idx <= HORIZON) & same_day

# causal features, per day (no cross-day leakage)
df["log_rate"] = np.log1p(df.rate); g = df.groupby("date", group_keys=False).log_rate
for w in (5, 15, 30, 60): df[f"roll_mean_{w}"] = g.transform(lambda s: s.rolling(w, min_periods=1).mean())
df["roll_max_15"] = g.transform(lambda s: s.rolling(15, min_periods=1).max())
df["roll_std_15"] = g.transform(lambda s: s.rolling(15, min_periods=1).std()).fillna(0)
df["roll_std_60"] = g.transform(lambda s: s.rolling(60, min_periods=1).std()).fillna(0)
df["bg_excess"] = df.log_rate - g.transform(lambda s: s.rolling(360, min_periods=30).quantile(0.10)).bfill()
df["rise_5"] = g.transform(lambda s: s.diff(5)).fillna(0); df["rise_15"] = g.transform(lambda s: s.diff(15)).fillna(0)
df["ev_id"] = -1
for i, (a, b) in enumerate(events): df.iloc[a:b + 1, df.columns.get_loc("ev_id")] = i
df["onset_id"] = -1  # onset event id of the flare that starts within the next 30 min
for i, s in enumerate(starts):
    lo = max(s - HORIZON, 0); df.iloc[lo:s, df.columns.get_loc("onset_id")] = i
df.loc[~df.y_pred, "onset_id"] = -1

def tss(y, yhat):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return tp / max(tp + fn, 1) - fp / max(fp + tn, 1)

def runs_of(b):
    r, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j + 1 < n and b[j + 1]: j += 1
            r.append((i, j)); i = j + 1
        else: i += 1
    return r

def evaluate(y, p, thr, ids, days, nboot=1000):
    yhat = p >= thr; ev = set(ids[ids >= 0]); det = set(ids[(ids >= 0) & yhat])
    rr = runs_of(yhat); false_runs = sum(1 for a, b in rr if not (ids[a:b + 1] >= 0).any())
    out = {"roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
           "pr_auc": average_precision_score(y, p) if y.sum() else float("nan"),
           "tss": tss(y, yhat), "n_events": len(ev), "events_detected": len(det),
           "event_recall": len(det) / len(ev) if ev else float("nan"),
           "alert_runs": len(rr), "false_alert_runs": false_runs}
    rng = np.random.default_rng(SEED); ud = np.unique(days); aucs = []
    for _ in range(nboot):
        pick = rng.choice(ud, len(ud)); ix = np.concatenate([np.where(days == d)[0] for d in pick])
        if 0 < y[ix].sum() < len(ix): aucs.append(roc_auc_score(y[ix], p[ix]))
    out["roc_auc_ci95"] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    return out

def latency(p, thr, starts_te):  # minutes from flare start to first alert (negative = alerted before start)
    lat = []
    for a, b in starts_te:
        hit = np.where(p[a:b + 1] >= thr)[0]
        if len(hit): lat.append(int(hit[0]))
    return lat

def best_thr(y, p):
    g = np.quantile(p, np.linspace(0.5, 0.999, 200)); return float(g[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in g]))])

results, test_preds = {}, {}
for task, ycol, idcol in (("NOWCAST (flare in progress)", "y_now", "ev_id"),
                          (f"FORECAST (flare onset in next {HORIZON} min)", "y_pred", "onset_id")):
    tr, va, te = (df[df.date.isin(s)] for s in (TRAIN, VAL, TEST))
    ytr, yva, yte = tr[ycol].to_numpy(int), va[ycol].to_numpy(int), te[ycol].to_numpy(int)
    Xtr, Xva, Xte = tr[FEATURES].to_numpy(), va[FEATURES].to_numpy(), te[FEATURES].to_numpy()
    ids, days = te[idcol].to_numpy(), te.date.to_numpy()
    print(f"\n{task}: train {len(tr)} (pos {ytr.sum()}), val {len(va)} (pos {yva.sum()}), test {len(te)} (pos {yte.sum()}, "
          f"events {len(set(ids[ids>=0]))})")
    rng = np.random.default_rng(SEED); res = {}; preds = {}
    scores = {"random": (rng.random(len(yte)), 0.5), "climatology": (np.full(len(yte), ytr.mean()), 0.5),
              # persistence: was a flare in progress one minute ago (benchmark convention)
              "persistence": (te.groupby("date").y_now.shift(1).fillna(False).to_numpy(float), 0.5)}
    lr_tr = tr.log_rate.to_numpy(); grid = np.quantile(lr_tr, np.linspace(0.5, 0.999, 200))
    thr_rate = grid[int(np.argmax([f1_score(ytr, lr_tr >= t, zero_division=0) for t in grid]))]
    scores["threshold_rate"] = (te.log_rate.to_numpy(), float(thr_rate))
    models = {"logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
              "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=30,
                                                      class_weight="balanced", random_state=SEED, n_jobs=-1),
              "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                                   class_weight="balanced", random_state=SEED)}
    from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
    for name, m in models.items():
        if name == "logistic": m = make_pipeline(StandardScaler(), m)
        m.fit(Xtr, ytr); scores[name] = (m.predict_proba(Xte)[:, 1], best_thr(yva, m.predict_proba(Xva)[:, 1]))
    for name, (p, thr) in scores.items():
        res[name] = evaluate(yte, p, thr, ids, days); res[name]["threshold"] = float(thr)
        if idcol == "ev_id":
            st = [(int(np.where(ids == e)[0][0]), int(np.where(ids == e)[0][-1])) for e in sorted(set(ids[ids >= 0]))]
            lat = latency(p, thr, st); res[name]["median_latency_min"] = float(np.median(lat)) if lat else None
        r = res[name]; print(f"  {name:16s} AUC={r['roc_auc']:.3f} CI={r['roc_auc_ci95'][0]:.2f}-{r['roc_auc_ci95'][1]:.2f} "
                             f"TSS={r['tss']:.2f} ev_recall={r['events_detected']}/{r['n_events']} false_runs={r['false_alert_runs']}/{r['alert_runs']}")
    if ycol == "y_pred":  # control from the AdityaNet report: quiet-only (no flare in progress) minutes
        q = ~te.y_now.to_numpy(); res["_quiet_only_auc"] = {n: float(roc_auc_score(yte[q], s[0][q])) for n, s in scores.items()
                                                           if 0 < yte[q].sum() < q.sum() and n not in ("random", "climatology")}
        print("  quiet-only AUC:", {k: round(v, 3) for k, v in res["_quiet_only_auc"].items()})
    results[task] = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_positives": int(yte.sum()),
                     "test_events": len(set(ids[ids >= 0])), "models": res}
    test_preds[task] = {n: s[0] for n, s in scores.items()}; test_preds[task]["_thr"] = {n: s[1] for n, s in scores.items()}

json.dump({"label_sigma": sigma, "n_events_total": len(events), "features": FEATURES, "split": {"train": TRAIN, "val": VAL, "test": TEST},
           "results": results}, open(f"{OUT}/results.json", "w"), indent=1)
df.drop(columns=["ev_id", "onset_id"]).to_csv(f"{OUT}/minute_dataset.csv")

# figure: test period, light curve with true flares and alerts of the best-F1-style models
te = df[df.date.isin(TEST)]; t = te.index
fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
ax[0].semilogy(t, te.rate, lw=.5); ax[0].set_ylabel("SoLEXS cts/s")
ax[0].fill_between(t, te.rate.min(), te.rate.max(), where=te.y_now.to_numpy(), color="red", alpha=.25, label="flare (label)"); ax[0].legend(loc="upper left")
nk = "NOWCAST (flare in progress)"; fk = f"FORECAST (flare onset in next {HORIZON} min)"
ax[1].plot(t, test_preds[nk]["random_forest"], lw=.6, label="random forest P(flare now)")
ax[1].axhline(test_preds[nk]["_thr"]["random_forest"], color="k", ls="--", lw=.6); ax[1].legend(loc="upper left"); ax[1].set_title("Nowcast")
ax[2].plot(t, test_preds[fk]["random_forest"], lw=.6, color="C1", label=f"random forest P(onset in {HORIZON} min)")
ax[2].fill_between(t, 0, 1, where=te.y_pred.to_numpy(), color="green", alpha=.25, label="true pre-onset window")
ax[2].axhline(test_preds[fk]["_thr"]["random_forest"], color="k", ls="--", lw=.6); ax[2].legend(loc="upper left"); ax[2].set_title("Forecast")
plt.tight_layout(); plt.savefig(f"{OUT}/test_period.png", dpi=80)
print("\nwritten:", OUT)
