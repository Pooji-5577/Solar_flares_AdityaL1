"""Upload a SoLEXS light-curve CSV -> flare nowcast (works) + 30-min forecast (near chance, see warning).
Run:  ../.venv/bin/streamlit run dashboard.py
Models are trained on the 11 real SoLEXS days (SDD2 total count rate only, so any rate CSV can be scored)."""
import os, sys
import numpy as np, pandas as pd
import streamlit as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from solex_common import load_minutes, label_events

H = 30  # forecast horizon, minutes

def features(rate: pd.Series) -> pd.DataFrame:
    """Causal features from a 1-min rate series (per UTC day), same as the 'base' set in run_strong.py."""
    d = pd.DataFrame({"rate": rate}); d["date"] = d.index.strftime("%Y-%m-%d")
    d["lr"] = np.log1p(d.rate); g = lambda f: d.groupby("date", group_keys=False)["lr"].transform(f)
    F = {}
    for w in (5, 15, 30, 60, 120, 240): F[f"mean_{w}"] = g(lambda s: s.rolling(w, min_periods=1).mean())
    for w in (15, 60, 180): F[f"max_{w}"] = g(lambda s: s.rolling(w, min_periods=1).max())
    for w in (15, 60): F[f"std_{w}"] = g(lambda s: s.rolling(w, min_periods=1).std()).fillna(0)
    bg = g(lambda s: s.rolling(360, min_periods=30).quantile(0.10)).bfill()
    ex = d.lr - bg; F["lr"], F["bg_excess"] = d.lr, ex
    for w in (60, 180): F[f"bg_excess_max{w}"] = ex.groupby(d.date).transform(lambda s: s.rolling(w, min_periods=1).max())
    for k in (3, 5, 15, 30): F[f"rise_{k}"] = g(lambda s: s.diff(k)).fillna(0)
    F["accel_5"] = F["rise_5"] - g(lambda s: s.diff(5).shift(5)).fillna(0)
    return pd.DataFrame(F, index=d.index).fillna(0)

def pick_thr(y, p):
    q = np.quantile(p, np.linspace(0.5, 0.999, 200))
    return float(q[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in q]))])

@st.cache_resource(show_spinner="Training on the 11 real SoLEXS days (one-time)...")
def train():
    df = load_minutes(); df["date"] = df.index.strftime("%Y-%m-%d")
    y_now, events, _ = label_events(df)
    starts = np.array([a for a, _ in events]); idx = np.arange(len(df)); nxt = np.searchsorted(starts, idx, side="right")
    ns = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], 10**9)
    same = np.array([df.date.iloc[min(s, len(df) - 1)] == dd for s, dd in zip(ns, df.date)])
    y_fc = (ns - idx <= H) & same
    X = features(df.rate).to_numpy(); days = df.date.to_numpy(); out = {}
    for name, y, C in (("nowcast", y_now, 0.1), ("forecast", y_fc, 0.01)):
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))
        oof = np.zeros(len(y))
        for dd in np.unique(days):  # leave-one-day-out so the alert threshold is not tuned on its own training data
            v = days == dd; oof[v] = mk().fit(X[~v], y[~v]).predict_proba(X[v])[:, 1]
        out[name] = (mk().fit(X, y), pick_thr(y, oof))
    return out, len(events), len(df)

def read_csv(file):
    raw = pd.read_csv(file)
    cols = {c.lower().strip(): c for c in raw.columns}
    tcol = next((cols[c] for c in ("time", "timestamp", "datetime", "date", "utc") if c in cols), raw.columns[0])
    rcol = next((cols[c] for c in ("rate", "counts", "count_rate", "cts", "flux", "counts_per_s") if c in cols), raw.columns[1])
    t = raw[tcol]
    t = pd.to_datetime(t, unit="s", utc=True) if pd.api.types.is_numeric_dtype(t) else pd.to_datetime(t, utc=True)
    s = pd.Series(pd.to_numeric(raw[rcol], errors="coerce").to_numpy(float), index=pd.DatetimeIndex(t)).sort_index()
    return s.resample("1min").mean(), tcol, rcol

def runs(b):
    r, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j + 1 < len(b) and b[j + 1]: j += 1
            r.append((i, j)); i = j + 1
        else: i += 1
    return r

st.set_page_config(page_title="SoLEXS flare predictor", layout="wide")
st.title("☀️ Solar flare nowcast & forecast (Aditya-L1 SoLEXS)")
st.caption("Upload a CSV with a time column and an X-ray count-rate column (counts/s). Rows can be 1-second or 1-minute.")
models, n_ev, n_min = train()
st.sidebar.markdown(f"**Trained on:** 11 real SoLEXS days, {n_min:,} minutes, {n_ev} flares.")
st.sidebar.markdown("**Nowcast** (is a flare happening now?): AUC 0.985 on 5 unseen days.\n\n"
                    "**Forecast** (flare starts in next 30 min?): AUC ≈ 0.5 on unseen days, i.e. **no skill yet**.")
f = st.file_uploader("CSV file", type="csv")
if f is None:
    st.info("Waiting for a CSV. Ready-made examples are in `solex_analysis/sample_inputs/`.")
    st.stop()

try:
    rate, tcol, rcol = read_csv(f)
except Exception as e:
    st.error(f"Could not read the file: {e}. Need a time column and a count-rate column."); st.stop()
if rate.notna().sum() < 30:
    st.error("Fewer than 30 valid minutes of data; need at least ~30 minutes."); st.stop()
st.success(f"Read **{rate.notna().sum():,} minutes** from {rate.index[0]:%Y-%m-%d %H:%M} to {rate.index[-1]:%Y-%m-%d %H:%M} UTC "
           f"(time = `{tcol}`, rate = `{rcol}`).")
rate = rate.interpolate(limit=10).fillna(rate.median())
X = features(rate).to_numpy(); res = pd.DataFrame({"rate": rate})
for name in ("nowcast", "forecast"):
    m, thr = models[name]; res[name + "_prob"] = m.predict_proba(X)[:, 1]; res[name + "_alert"] = res[name + "_prob"] >= thr
_, ref_events, _ = label_events(res[["rate"]])
na, fa = runs(res.nowcast_alert.to_numpy()), runs(res.forecast_alert.to_numpy())

c1, c2, c3 = st.columns(3)
c1.metric("Flare alerts (nowcast)", len(na)); c2.metric("Minutes flagged as flaring", int(res.nowcast_alert.sum()))
c3.metric("Forecast warnings (unreliable)", len(fa))

t = res.index; fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
ax[0].semilogy(t, res.rate, lw=.6, color="#1f4e79"); ax[0].set_ylabel("counts/s"); ax[0].set_title("X-ray light curve")
ax[1].plot(t, res.nowcast_prob, lw=.7, color="#c0392b"); ax[1].axhline(models["nowcast"][1], ls="--", color="k", lw=.6)
ax[1].fill_between(t, 0, 1, where=res.nowcast_alert, color="red", alpha=.2); ax[1].set_ylabel("probability"); ax[1].set_title("NOWCAST: is a flare happening now? (dashed = alert threshold)")
ax[2].plot(t, res.forecast_prob, lw=.7, color="#d68910"); ax[2].axhline(models["forecast"][1], ls="--", color="k", lw=.6)
ax[2].fill_between(t, 0, 1, where=res.forecast_alert, color="orange", alpha=.2); ax[2].set_ylabel("probability"); ax[2].set_title("FORECAST: flare in next 30 min? (experimental, near chance)")
for a in ax: a.margins(x=0)
plt.tight_layout(); st.pyplot(fig)

st.subheader("Detected flare periods (nowcast)")
if na:
    st.dataframe(pd.DataFrame([{"start (UTC)": t[a], "end (UTC)": t[b], "minutes": b - a + 1, "peak counts/s": round(float(res.rate.iloc[a:b + 1].max()), 1),
                                "peak probability": round(float(res.nowcast_prob.iloc[a:b + 1].max()), 3)} for a, b in na]), use_container_width=True)
else:
    st.write("No flare periods detected.")
st.caption(f"For comparison, the simple reference detector (non-causal background rule) finds {len(ref_events)} flares in this file.")
st.warning("**Forecast caveat:** on 5 held-out days the 30-minute forecast scored AUC ≈ 0.5 (chance), because only ~26 flare onsets were available to learn from. "
           "Treat orange warnings as experimental, not as a prediction you can rely on.")
st.download_button("Download results CSV", res.reset_index().rename(columns={"index": "time"}).to_csv(index=False), "solex_predictions.csv", "text/csv")
