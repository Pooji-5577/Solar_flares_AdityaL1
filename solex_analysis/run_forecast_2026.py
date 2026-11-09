"""Forecast/nowcast experiment on the 2026 SoLEXS days (32 days: Feb, Apr, Jun and 26 days of Oct 2026).

Same protocol as run_strong.py (causal base features, detector labels, 30-min onset target) with two changes made for
a larger, gappy dataset: 5 time-ordered blocked folds instead of leave-one-day-out (speed), and base features only
(no PI spectrum). Train = earlier days, test = last 10 days, scored once. Writes to results_2026/ only.
"""
import itertools, json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import solex_common as sc
warnings.filterwarnings("ignore")
sc.DATA = os.path.join(HERE, "..", "solex_data", "data_2026")
OUT = os.path.join(HERE, "results_2026"); os.makedirs(OUT, exist_ok=True)
SEED, H, N_TEST_DAYS, N_FOLDS = 20260718, 30, 10, 5

df = sc.load_minutes(); df["date"] = df.index.strftime("%Y-%m-%d")
flare, events, sigma = sc.label_events(df, k=4.0)          # labels from the raw series (NaN where no data)
df["y_now"] = flare
df["rate_f"] = df.groupby("date").rate.transform(lambda s: s.ffill(limit=10))   # bridge gaps up to 10 min
valid = df.rate_f.notna().to_numpy()
starts = np.array([a for a, _ in events]); idx = np.arange(len(df)); nxt = np.searchsorted(starts, idx, side="right")
ns = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], 10**9)
sd = np.array([df.date.iloc[min(s, len(df) - 1)] == d for s, d in zip(ns, df.date)])
df["y_pred"] = (ns - idx <= H) & sd
ev_id = np.full(len(df), -1); on_id = np.full(len(df), -1)
for i, (a, b) in enumerate(events): ev_id[a:b + 1] = i; on_id[max(a - H, 0):a] = i
on_id[~df.y_pred.to_numpy()] = -1

# ---- causal base features (identical definitions to run_strong.py) ----
df["log_rate"] = np.log1p(df.rate_f); g = lambda c: df.groupby("date", group_keys=False)[c]
F = {}
for w in (5, 15, 30, 60, 120, 240): F[f"mean_{w}"] = g("log_rate").transform(lambda s: s.rolling(w, min_periods=1).mean())
for w in (15, 60, 180): F[f"max_{w}"] = g("log_rate").transform(lambda s: s.rolling(w, min_periods=1).max())
for w in (15, 60): F[f"std_{w}"] = g("log_rate").transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
bg = g("log_rate").transform(lambda s: s.rolling(360, min_periods=30).quantile(0.10)).bfill()
F["log_rate"], F["bg_excess"] = df.log_rate, df.log_rate - bg
be = pd.Series(F["bg_excess"].to_numpy(), index=df.index)
F["bg_excess_max60"] = be.groupby(df.date).transform(lambda s: s.rolling(60, min_periods=1).max())
F["bg_excess_max180"] = be.groupby(df.date).transform(lambda s: s.rolling(180, min_periods=1).max())
for k in (3, 5, 15, 30): F[f"rise_{k}"] = g("log_rate").transform(lambda s: s.diff(k)).fillna(0)
F["accel_5"] = F["rise_5"] - g("log_rate").transform(lambda s: s.diff(5).shift(5)).fillna(0)
X_all = pd.DataFrame(F).fillna(0); assert X_all.shape[1] == 20

days = sorted(df.date.unique()); TEST_DAYS, TRAIN_DAYS = days[-N_TEST_DAYS:], days[:-N_TEST_DAYS]
tr_mask = (df.date.isin(TRAIN_DAYS).to_numpy()) & valid; te_mask = (df.date.isin(TEST_DAYS).to_numpy()) & valid
fold_days = [list(a) for a in np.array_split(TRAIN_DAYS, N_FOLDS)]
print(f"days {len(days)} | flares {len(events)} | train days {len(TRAIN_DAYS)} | test days {len(TEST_DAYS)} ({TEST_DAYS[0]}..{TEST_DAYS[-1]})")
print("test flares:", len({i for i in ev_id[te_mask] if i >= 0}), "| dropped (no-data) minutes:", int((~valid).sum()), flush=True)

def make(name, hp):
    if name == "logistic": return make_pipeline(StandardScaler(), LogisticRegression(C=hp, max_iter=4000, class_weight="balanced"))
    if name == "random_forest": return RandomForestClassifier(n_estimators=200, max_depth=hp[0], min_samples_leaf=hp[1], class_weight="balanced", random_state=SEED, n_jobs=-1)
    return HistGradientBoostingClassifier(max_depth=hp[0], learning_rate=0.05, max_iter=hp[1], l2_regularization=hp[2], class_weight="balanced", random_state=SEED)
GRID = {"logistic": [0.01, 0.1, 1.0], "random_forest": [(6, 50), (12, 30)], "gradient_boosting": [(3, 150, 1.0), (4, 200, 1.0), (3, 300, 10.0)]}

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
def best_thr(y, p):
    q = np.quantile(p, np.linspace(0.5, 0.999, 200)); return float(q[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in q]))])
def rank01(p, ref): return np.searchsorted(np.sort(ref), p) / len(ref)

dates = df.date.to_numpy(); results = {}; test_scores = {}
X = X_all.to_numpy()
for task, ycol, idarr in (("NOWCAST", "y_now", ev_id), ("FORECAST", "y_pred", on_id)):
    y = df[ycol].to_numpy(bool); ytr, yte = y[tr_mask], y[te_mask]; Xtr, Xte = X[tr_mask], X[te_mask]; dtr = dates[tr_mask]
    print(f"\n=== {task}: train {tr_mask.sum()} (pos {ytr.sum()}), test {te_mask.sum()} (pos {yte.sum()}) ===", flush=True)
    cv = {}
    for name in GRID:
        for hp in GRID[name]:
            oof = np.zeros(len(ytr)); sc_ = []
            for fd in fold_days:
                v = np.isin(dtr, fd); m = make(name, hp).fit(Xtr[~v], ytr[~v]); oof[v] = m.predict_proba(Xtr[v])[:, 1]
                if 0 < ytr[v].sum() < v.sum(): sc_.append(average_precision_score(ytr[v], oof[v]))
            cv[(name, str(hp))] = (float(np.mean(sc_)), oof, hp)
    best = {n: max((k for k in cv if k[0] == n), key=lambda k: cv[k][0]) for n in GRID}
    members, scores, oof_ens = {}, {}, np.zeros(len(ytr))
    for n, k in best.items():
        print(f"  CV best {n:18s} hp={k[1]:14s} CV PR-AUC={cv[k][0]:.3f}", flush=True)
        m = make(n, cv[k][2]).fit(Xtr, ytr); members[n] = (m, np.sort(cv[k][1])); oof_ens += rank01(cv[k][1], np.sort(cv[k][1])) / len(best)
        scores[n] = (m.predict_proba(Xte)[:, 1], best_thr(ytr, cv[k][1]))
    scores["ENSEMBLE"] = (np.mean([rank01(scores[n][0], members[n][1]) for n in members], axis=0), best_thr(ytr, oof_ens))
    lr_tr = df.log_rate.to_numpy()[tr_mask]; q = np.quantile(lr_tr, np.linspace(0.5, 0.999, 200))
    scores["threshold_rate"] = (df.log_rate.to_numpy()[te_mask], float(q[int(np.argmax([f1_score(ytr, lr_tr >= t, zero_division=0) for t in q]))]))
    scores["persistence"] = (df.groupby("date").y_now.shift(1).fillna(False).to_numpy(float)[te_mask], 0.5)
    ids, dte = idarr[te_mask], dates[te_mask]; res = {}
    for n, (p, thr) in scores.items():
        yh = p >= thr; ev = set(ids[ids >= 0]); det = set(ids[(ids >= 0) & yh]); rr = runs_of(yh)
        fr = sum(1 for a, b in rr if not (ids[a:b + 1] >= 0).any()); rng = np.random.default_rng(SEED); ud = np.unique(dte); au = []
        for _ in range(1000):
            ix = np.concatenate([np.where(dte == d)[0] for d in rng.choice(ud, len(ud))])
            if 0 < yte[ix].sum() < len(ix): au.append(roc_auc_score(yte[ix], p[ix]))
        res[n] = {"roc_auc": float(roc_auc_score(yte, p)), "roc_auc_ci95": [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
                  "pr_auc": float(average_precision_score(yte, p)), "tss": float(tss(yte, yh)), "n_events": len(ev), "events_detected": len(det),
                  "alert_runs": len(rr), "false_alert_runs": fr, "threshold": float(thr)}
        r = res[n]; print(f"  {n:18s} AUC={r['roc_auc']:.3f} [{r['roc_auc_ci95'][0]:.2f},{r['roc_auc_ci95'][1]:.2f}] PR={r['pr_auc']:.3f} TSS={r['tss']:.2f} "
                          f"events={r['events_detected']}/{r['n_events']} false_runs={fr}/{len(rr)}", flush=True)
    if task == "FORECAST":
        qm = ~df.y_now.to_numpy()[te_mask]
        res["_quiet_only_auc"] = {n: float(roc_auc_score(yte[qm], s[0][qm])) for n, s in scores.items() if n != "persistence"}
        print("  quiet-only AUC:", {k: round(v, 3) for k, v in res["_quiet_only_auc"].items()})
    results[task] = {"selected": {n: {"hp": k[1], "cv_pr_auc": cv[k][0]} for n, k in best.items()}, "base_rate_test": float(yte.mean()), "models": res}
    test_scores[task] = scores
ts = pd.DataFrame({f"{t}_{n}": v[0] for t, sc2 in test_scores.items() for n, v in sc2.items()}, index=df.index[te_mask])
ts["y_now"] = df.y_now.to_numpy()[te_mask]; ts["y_pred"] = df.y_pred.to_numpy()[te_mask]; ts.to_csv(f"{OUT}/test_scores_2026.csv")
json.dump({"days": days, "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "n_events": len(events), "label_sigma": sigma,
           "n_features": 20, "folds": N_FOLDS, "results": results}, open(f"{OUT}/results_2026.json", "w"), indent=1)
print("\nwritten", OUT)
