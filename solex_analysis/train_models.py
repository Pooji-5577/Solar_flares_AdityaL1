"""Train nowcasting AND forecasting models from dataset/training_table.csv (built by build_dataset.py).

Reads the table only: no FITS, no feature building here. Protocol (same as run_strong.py / run_forecast_combined.py):
chronological split (the table's `split` column), 5 time-ordered blocked folds on the training days to pick
hyperparameters and the feature set (base 20 vs base+spectral 48) by PR-AUC, class-balanced logistic / random forest /
gradient boosting + rank-average ensemble, alert threshold chosen on training data, test days scored once.

usage: python train_models.py [auto|base]     (base = 20 base features only, no spectral)
Writes results_table/ : results_table.json, test_scores_table.csv, models_nowcast.joblib, models_forecast.joblib
"""
import json, os, sys, warnings
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "dataset", "training_table.csv"); SEED, N_FOLDS = 20260718, 5
MODE = sys.argv[1] if len(sys.argv) > 1 else "auto"
OUT = os.path.join(HERE, "results_table" if MODE == "auto" else "results_table_base"); os.makedirs(OUT, exist_ok=True)

t = pd.read_csv(CSV, index_col=0, parse_dates=True, low_memory=False)
cols = list(t.columns)
BASE = cols[cols.index("y_forecast_30min") + 1: cols.index("lb0")]; SPEC = cols[cols.index("lb0"): cols.index("band0")]
assert len(BASE) == 20 and len(SPEC) == 28, (len(BASE), len(SPEC))
FSETS = {"base": BASE} if MODE == "base" else {"base": BASE, "base+spectral": BASE + SPEC}
valid = t.has_data.to_numpy(bool); dates = t.date.to_numpy()
tr_mask = (t.split.to_numpy() == "train") & valid; te_mask = (t.split.to_numpy() == "test") & valid
X_all = {fs: t[c].fillna(0).to_numpy(float) for fs, c in FSETS.items()}

# event ids rebuilt from the labels: a run of y_now=1 (unbroken in time) is one flare; forecast minutes point at the next flare
y_now = t.y_now.to_numpy(bool); gap = np.r_[True, np.diff(t.index.values).astype("timedelta64[s]").astype(int) > 60]
start = y_now & (~np.r_[False, y_now[:-1]] | gap); run = np.cumsum(start) - 1
ev_id = np.where(y_now, run, -1); starts = np.where(start)[0]
nxt = np.searchsorted(starts, np.arange(len(t)), side="right")
on_id = np.where(t.y_forecast_30min.to_numpy(bool) & (nxt < len(starts)), nxt, -1)
print(f"table {len(t)} rows | {len(starts)} flares | train {tr_mask.sum()} rows / test {te_mask.sum()} rows | feature sets: {list(FSETS)}")

train_days = sorted(set(dates[tr_mask])); fold_days = [list(a) for a in np.array_split(train_days, N_FOLDS)]
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
    q = np.quantile(p, np.linspace(0.5, 0.999, 200)); return float(q[int(np.argmax([f1_score(y, p >= v, zero_division=0) for v in q]))])
def rank01(p, ref): return np.searchsorted(np.sort(ref), p) / len(ref)

results, all_scores = {}, {}
for task, ycol, idarr in (("NOWCAST", "y_now", ev_id), ("FORECAST", "y_forecast_30min", on_id)):
    y = t[ycol].to_numpy(bool); ytr, yte = y[tr_mask], y[te_mask]; dtr = dates[tr_mask]
    print(f"\n=== {task}: train {tr_mask.sum()} (pos {ytr.sum()}), test {te_mask.sum()} (pos {yte.sum()}) ===", flush=True)
    cv = {}
    for fs in FSETS:
        Xtr = X_all[fs][tr_mask]
        for name in GRID:
            for hp in GRID[name]:
                oof = np.zeros(len(ytr)); sc_ = []
                for fd in fold_days:
                    v = np.isin(dtr, fd); m = make(name, hp).fit(Xtr[~v], ytr[~v]); oof[v] = m.predict_proba(Xtr[v])[:, 1]
                    if 0 < ytr[v].sum() < v.sum(): sc_.append(average_precision_score(ytr[v], oof[v]))
                cv[(fs, name, str(hp))] = (float(np.mean(sc_)), oof, hp)
    best = {n: max((k for k in cv if k[1] == n), key=lambda k: cv[k][0]) for n in GRID}
    members, scores, oof_ens, saved = {}, {}, np.zeros(len(ytr)), {}
    for n, k in best.items():
        print(f"  CV best {n:18s} feats={k[0]:14s} hp={k[2]:14s} CV PR-AUC={cv[k][0]:.3f}", flush=True)
        m = make(n, cv[k][2]).fit(X_all[k[0]][tr_mask], ytr); ref = np.sort(cv[k][1]); members[n] = (m, ref, k[0])
        oof_ens += rank01(cv[k][1], ref) / len(best)
        scores[n] = (m.predict_proba(X_all[k[0]][te_mask])[:, 1], best_thr(ytr, cv[k][1]))
        saved[n] = {"model": m, "feature_set": k[0], "features": FSETS[k[0]], "oof_ref": ref, "threshold": scores[n][1], "hp": k[2]}
    scores["ENSEMBLE"] = (np.mean([rank01(scores[n][0], members[n][1]) for n in members], axis=0), best_thr(ytr, oof_ens))
    lr_tr = t.log_rate.to_numpy()[tr_mask]; q = np.quantile(lr_tr, np.linspace(0.5, 0.999, 200))
    scores["threshold_rate"] = (t.log_rate.to_numpy()[te_mask], float(q[int(np.argmax([f1_score(ytr, lr_tr >= v, zero_division=0) for v in q]))]))
    scores["persistence"] = (t.groupby("date").y_now.shift(1).fillna(0).to_numpy(float)[te_mask], 0.5)
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
    results[task] = {"selected": {n: {"features": k[0], "hp": k[2], "cv_pr_auc": cv[k][0]} for n, k in best.items()}, "base_rate_test": float(yte.mean()), "models": res}
    all_scores[task] = scores
    joblib.dump({"task": task, "members": saved, "ensemble_threshold": scores["ENSEMBLE"][1], "train_days": train_days,
                 "note": "ensemble score = mean rank of member probabilities against each member's out-of-fold reference"},
                os.path.join(OUT, f"models_{task.lower()}.joblib"))
ts = pd.DataFrame({f"{tk}_{n}": v[0] for tk, sc2 in all_scores.items() for n, v in sc2.items()}, index=t.index[te_mask])
ts["y_now"] = t.y_now.to_numpy()[te_mask]; ts["y_forecast_30min"] = t.y_forecast_30min.to_numpy()[te_mask]; ts.to_csv(os.path.join(OUT, "test_scores_table.csv"))
json.dump({"mode": MODE, "n_flares": int(len(starts)), "folds": N_FOLDS, "test_days": sorted(set(dates[te_mask])), "results": results},
          open(os.path.join(OUT, "results_table.json"), "w"), indent=1)
print("\nwritten", OUT)
