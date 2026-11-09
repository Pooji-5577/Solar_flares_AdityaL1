"""Stronger SoLEXS nowcast/forecast: spectral + recent-activity features, grouped-CV tuning, ensemble.
Tuning and thresholds use ONLY the train days (20-26 Mar); test days (27-31 Mar) are scored once, same as run_analysis.py."""
import glob, gzip, io, itertools, json, os, sys, warnings, zipfile
import numpy as np, pandas as pd
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
sys.path.insert(0, os.path.dirname(__file__))
from solex_common import DATA, load_minutes, label_events
warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "results"); SEED, H = 20260718, 30
TRAIN_DAYS = ["2025-03-20", "2025-03-22", "2025-03-23", "2025-03-24", "2025-03-25", "2025-03-26"]
TEST_DAYS = ["2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31"]
BANDS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 340)]

# ---- data: 1-min total rate + 6 spectral band rates ----
def load_bands():
    cache = f"{OUT}/bands.csv"
    if os.path.exists(cache): return pd.read_csv(cache, index_col=0, parse_dates=True)
    frames = []
    for z in sorted(glob.glob(os.path.join(DATA, "*.zip"))):
        zf = zipfile.ZipFile(z); n = [x for x in zf.namelist() if x.endswith("SDD2_L1.pi.gz")][0]
        h = fits.open(io.BytesIO(gzip.decompress(zf.read(n))))[1].data
        c = np.nan_to_num(np.asarray(h["COUNTS"], float)).reshape(1440, 60, 340).sum(1) / 60
        t0 = pd.to_datetime(float(h["TSTART"][0]), unit="s", utc=True)
        frames.append(pd.DataFrame({f"b{i}": c[:, a:b].sum(1) for i, (a, b) in enumerate(BANDS)},
                                   index=pd.date_range(t0, periods=1440, freq="1min")))
    d = pd.concat(frames); d.to_csv(cache); return d

df = load_minutes().join(load_bands(), how="left"); df["date"] = df.index.strftime("%Y-%m-%d")
flare, events, _ = label_events(df, k=4.0); df["y_now"] = flare
starts = np.array([a for a, _ in events]); idx = np.arange(len(df)); nxt = np.searchsorted(starts, idx, side="right")
ns = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], 10**9)
sd = np.array([df.date.iloc[min(s, len(df) - 1)] == d for s, d in zip(ns, df.date)])
df["y_pred"] = (ns - idx <= H) & sd
ev_id = np.full(len(df), -1); on_id = np.full(len(df), -1)
for i, (a, b) in enumerate(events): ev_id[a:b + 1] = i
for i, (a, b) in enumerate(events): on_id[max(a - H, 0):a] = i
on_id[~df.y_pred.to_numpy()] = -1; df["ev_id"], df["on_id"] = ev_id, on_id

# ---- features (causal, per day) ----
df["log_rate"] = np.log1p(df.rate); g = lambda c: df.groupby("date", group_keys=False)[c]
F = {}
for w in (5, 15, 30, 60, 120, 240): F[f"mean_{w}"] = g("log_rate").transform(lambda s: s.rolling(w, min_periods=1).mean())
for w in (15, 60, 180): F[f"max_{w}"] = g("log_rate").transform(lambda s: s.rolling(w, min_periods=1).max())
for w in (15, 60): F[f"std_{w}"] = g("log_rate").transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
bg = g("log_rate").transform(lambda s: s.rolling(360, min_periods=30).quantile(0.10)).bfill()
F["log_rate"], F["bg_excess"] = df.log_rate, df.log_rate - bg
F["bg_excess_max60"] = pd.Series(F["bg_excess"].to_numpy(), index=df.index).groupby(df.date).transform(lambda s: s.rolling(60, min_periods=1).max())
F["bg_excess_max180"] = pd.Series(F["bg_excess"].to_numpy(), index=df.index).groupby(df.date).transform(lambda s: s.rolling(180, min_periods=1).max())
for k in (3, 5, 15, 30): F[f"rise_{k}"] = g("log_rate").transform(lambda s: s.diff(k)).fillna(0)
F["accel_5"] = F["rise_5"] - g("log_rate").transform(lambda s: s.diff(5).shift(5)).fillna(0)
BASE = list(F)
for i in range(len(BANDS)): df[f"lb{i}"] = np.log1p(df[f"b{i}"])
for i in range(len(BANDS)):
    F[f"lb{i}"] = df[f"lb{i}"]; F[f"lb{i}_m5"] = g(f"lb{i}").transform(lambda s: s.rolling(5, min_periods=1).mean())
    F[f"lb{i}_rise5"] = g(f"lb{i}").transform(lambda s: s.diff(5)).fillna(0)
for i in range(1, len(BANDS)):  # hardness ratios (higher band / softest band), smoothed
    F[f"hard_{i}"] = F[f"lb{i}_m5"] - F["lb0_m5"]; F[f"hard_{i}_rise5"] = pd.Series(F[f"hard_{i}"].to_numpy(), index=df.index).groupby(df.date).diff(5).fillna(0)
X_all = pd.DataFrame(F); SPEC = [c for c in X_all if c not in BASE]
FSETS = {"base": BASE[:], "base+spectral": BASE + SPEC}
X_all = X_all.fillna(0)

def make(name, hp):
    if name == "logistic": return make_pipeline(StandardScaler(), LogisticRegression(C=hp, max_iter=4000, class_weight="balanced"))
    if name == "random_forest": return RandomForestClassifier(n_estimators=300, max_depth=hp[0], min_samples_leaf=hp[1], class_weight="balanced", random_state=SEED, n_jobs=-1)
    return HistGradientBoostingClassifier(max_depth=hp[0], learning_rate=0.05, max_iter=hp[1], l2_regularization=hp[2], class_weight="balanced", random_state=SEED)
GRID = {"logistic": [0.01, 0.1, 1.0], "random_forest": [(6, 50), (8, 30), (12, 30)],
        "gradient_boosting": [(3, 150, 1.0), (4, 200, 1.0), (3, 300, 10.0), (6, 150, 10.0)]}

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
    g_ = np.quantile(p, np.linspace(0.5, 0.999, 200)); return float(g_[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in g_]))])
def rank01(p, ref): return np.searchsorted(np.sort(ref), p) / len(ref)

tr_mask = df.date.isin(TRAIN_DAYS).to_numpy(); te_mask = df.date.isin(TEST_DAYS).to_numpy()
days_tr = df.date.to_numpy()[tr_mask]
results, test_scores = {}, {}
for task, ycol, idarr in (("NOWCAST", "y_now", ev_id), ("FORECAST", "y_pred", on_id)):
    y = df[ycol].to_numpy(bool); ytr, yte = y[tr_mask], y[te_mask]
    print(f"\n=== {task}: train {tr_mask.sum()} (pos {ytr.sum()}), test {te_mask.sum()} (pos {yte.sum()}) ===", flush=True)
    # 1. grouped-CV (leave-one-day-out over the 6 train days) model + feature-set selection
    cv, oof_store = {}, {}
    for fs, name in itertools.product(FSETS, GRID):
        Xtr = X_all.loc[:, FSETS[fs]].to_numpy()[tr_mask]
        for hp in GRID[name]:
            oof = np.zeros(len(ytr)); sc = []
            for d in TRAIN_DAYS:
                v = days_tr == d; m = make(name, hp).fit(Xtr[~v], ytr[~v]); oof[v] = m.predict_proba(Xtr[v])[:, 1]
                if 0 < ytr[v].sum() < v.sum(): sc.append(average_precision_score(ytr[v], oof[v]))
            cv[(fs, name, str(hp))] = (float(np.mean(sc)), oof, hp)
    best = {}
    for name in GRID:
        k = max((k for k in cv if k[1] == name), key=lambda k: cv[k][0]); best[name] = k
        print(f"  CV best {name:18s} feats={k[0]:14s} hp={k[2]:16s} CV PR-AUC={cv[k][0]:.3f}", flush=True)
    # 2. ensemble = mean of rank-normalised member scores (members fit on all train days)
    members, oof_ens = {}, np.zeros(len(ytr))
    for name, k in best.items():
        Xtr = X_all.loc[:, FSETS[k[0]]].to_numpy()[tr_mask]; Xte = X_all.loc[:, FSETS[k[0]]].to_numpy()[te_mask]
        m = make(name, cv[k][2]).fit(Xtr, ytr); members[name] = (m, k[0], np.sort(cv[k][1]))
        oof_ens += rank01(cv[k][1], np.sort(cv[k][1])) / len(best)
    scores = {}
    for name, (m, fs, ref) in members.items():
        Xte = X_all.loc[:, FSETS[fs]].to_numpy()[te_mask]; scores[name] = (m.predict_proba(Xte)[:, 1], best_thr(ytr, cv[best[name]][1]))
    ens = np.mean([rank01(scores[n][0], members[n][2]) for n in members], axis=0); scores["ENSEMBLE"] = (ens, best_thr(ytr, oof_ens))
    # 3. baselines + test scoring
    lr_tr = df.log_rate.to_numpy()[tr_mask]; grid = np.quantile(lr_tr, np.linspace(0.5, 0.999, 200))
    scores["threshold_rate"] = (df.log_rate.to_numpy()[te_mask], float(grid[int(np.argmax([f1_score(ytr, lr_tr >= t, zero_division=0) for t in grid]))]))
    scores["persistence"] = (df.groupby("date").y_now.shift(1).fillna(False).to_numpy(float)[te_mask], 0.5)
    ids, days_te = idarr[te_mask], df.date.to_numpy()[te_mask]; res = {}
    for name, (p, thr) in scores.items():
        yh = p >= thr; ev = set(ids[ids >= 0]); det = set(ids[(ids >= 0) & yh]); rr = runs_of(yh)
        fr = sum(1 for a, b in rr if not (ids[a:b + 1] >= 0).any()); rng = np.random.default_rng(SEED); ud = np.unique(days_te); au = []
        for _ in range(1000):
            ix = np.concatenate([np.where(days_te == d)[0] for d in rng.choice(ud, len(ud))])
            if 0 < yte[ix].sum() < len(ix): au.append(roc_auc_score(yte[ix], p[ix]))
        res[name] = {"roc_auc": float(roc_auc_score(yte, p)), "roc_auc_ci95": [float(np.percentile(au, 2.5)), float(np.percentile(au, 97.5))],
                     "pr_auc": float(average_precision_score(yte, p)), "tss": float(tss(yte, yh)), "n_events": len(ev), "events_detected": len(det),
                     "alert_runs": len(rr), "false_alert_runs": fr, "threshold": float(thr)}
        print(f"  {name:18s} AUC={res[name]['roc_auc']:.3f} [{res[name]['roc_auc_ci95'][0]:.2f},{res[name]['roc_auc_ci95'][1]:.2f}] PR={res[name]['pr_auc']:.3f} "
              f"TSS={res[name]['tss']:.2f} events={len(det)}/{len(ev)} false_runs={fr}/{len(rr)}", flush=True)
    if task == "FORECAST":
        q = ~df.y_now.to_numpy()[te_mask]; res["_quiet_only_auc"] = {n: float(roc_auc_score(yte[q], s[0][q])) for n, s in scores.items() if n != "persistence"}
        print("  quiet-only AUC:", {k: round(v, 3) for k, v in res["_quiet_only_auc"].items()})
    results[task] = {"selected": {n: {"features": k[0], "hp": k[2], "cv_pr_auc": cv[k][0]} for n, k in best.items()}, "models": res}
    test_scores[task] = scores
_ts = pd.DataFrame({f"{t}_{n}": v[0] for t, sc in test_scores.items() for n, v in sc.items()}, index=df.index[te_mask])
_ts["y_now"] = df.y_now.to_numpy()[te_mask]; _ts["y_pred"] = df.y_pred.to_numpy()[te_mask]; _ts.to_csv(f"{OUT}/test_scores_strong.csv")
json.dump({"thresholds": {t: {n: float(v[1]) for n, v in sc.items()} for t, sc in test_scores.items()}}, open(f"{OUT}/thresholds_strong.json", "w"))
json.dump({"n_features": {k: len(v) for k, v in FSETS.items()}, "results": results}, open(f"{OUT}/results_strong.json", "w"), indent=1)

t = df.index[te_mask]; fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
ax[0].semilogy(t, df.rate[te_mask], lw=.5); ax[0].fill_between(t, df.rate.min(), df.rate.max(), where=df.y_now.to_numpy()[te_mask], color="red", alpha=.25, label="flare (label)"); ax[0].legend(loc="upper left"); ax[0].set_ylabel("cts/s")
for a, task, c, yc in ((ax[1], "NOWCAST", "C0", "y_now"), (ax[2], "FORECAST", "C1", "y_pred")):
    p, thr = test_scores[task]["ENSEMBLE"]; a.plot(t, p, lw=.6, color=c, label=f"ensemble {task.lower()} score"); a.axhline(thr, color="k", ls="--", lw=.6)
    a.fill_between(t, 0, 1, where=df[yc].to_numpy()[te_mask], color="green" if task == "FORECAST" else "red", alpha=.2); a.legend(loc="upper left"); a.set_title(task)
plt.tight_layout(); plt.savefig(f"{OUT}/test_period_strong.png", dpi=80); print("\nwritten results_strong.json, test_period_strong.png")
