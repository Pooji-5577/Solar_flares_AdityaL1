"""Build the ML training table: raw SoLEXS FITS zips -> one row per minute, features + labels, for nowcasting and forecasting.

Output: dataset/training_table.csv (+ dataset/DATA_DICTIONARY.md). Covers all 50 days (event days 2024 + Mar 2025 + 2026).
Same definitions as run_strong.py / run_forecast_combined.py: causal features, detector labels (each period labelled
separately), forecast target = a flare onset within the next 30 minutes (same day).
"""
import glob, gzip, io, os, sys, warnings, zipfile
import numpy as np, pandas as pd
from astropy.io import fits
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import solex_common as sc
warnings.filterwarnings("ignore")
OUT = os.path.join(HERE, "dataset"); os.makedirs(OUT, exist_ok=True)
H, N_TEST_DAYS = 30, 10
PERIODS = [("events_2024", os.path.join(HERE, "..", "solex_data", "flare_event_data")), ("mar_2025", os.path.join(HERE, "..", "solex_data", "data")), ("year_2026", os.path.join(HERE, "..", "solex_data", "data_2026"))]
BANDS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 340)]   # PI channel index ranges (ordinal, not keV)

def load_bands(folder):
    frames = []
    for z in sorted(glob.glob(os.path.join(folder, "*.zip"))):
        zf = zipfile.ZipFile(z); name = [n for n in zf.namelist() if n.endswith("SDD2_L1.pi.gz")][0]
        d = fits.open(io.BytesIO(gzip.decompress(zf.read(name))))[1].data
        if len(d["COUNTS"]) != 86400: print("skip PI (rows != 86400):", os.path.basename(z)); continue
        c = np.nan_to_num(np.asarray(d["COUNTS"], float)).reshape(1440, 60, 340).sum(1) / 60      # counts/s per channel, per minute
        t0 = pd.to_datetime(float(d["TSTART"][0]), unit="s", utc=True).floor("D")
        frames.append(pd.DataFrame({f"band{i}": c[:, a:b].sum(1) for i, (a, b) in enumerate(BANDS)}, index=pd.date_range(t0, periods=1440, freq="1min")))
    return pd.concat(frames)

# ---- 1. read FITS -> 1-minute series, label flares (each period separately), add spectral bands ----
parts, events, off = [], [], 0
for tag, folder in PERIODS:
    sc.DATA = folder; p = sc.load_minutes(); fl, ev, sg = sc.label_events(p, k=4.0)
    p["y_now"] = fl; p["period"] = tag; events += [(a + off, b + off) for a, b in ev]; off += len(p)
    p = p.join(load_bands(folder), how="left"); parts.append(p); print(f"{tag}: {len(p)} minutes, {len(ev)} flares", flush=True)
df = pd.concat(parts); df["date"] = df.index.strftime("%Y-%m-%d")
df["rate_f"] = df.groupby("date").rate.transform(lambda s: s.ffill(limit=10))      # bridge gaps up to 10 min
df["valid"] = df.rate_f.notna()

# ---- 2. labels: nowcast (flare in progress) and forecast (onset in next 30 min, same day) ----
starts = np.array([a for a, _ in events]); idx = np.arange(len(df)); nxt = np.searchsorted(starts, idx, side="right")
ns = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)], 10**9)
same_day = np.array([df.date.iloc[min(s, len(df) - 1)] == d for s, d in zip(ns, df.date)])
df["y_forecast_30min"] = (ns - idx <= H) & same_day
df["y_now"] = df.y_now.astype(bool)

# ---- 3. causal features (past data only, per day) ----
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
BASE = list(F)
for i in range(len(BANDS)): df[f"lb{i}"] = np.log1p(df[f"band{i}"])
for i in range(len(BANDS)):
    F[f"lb{i}"] = df[f"lb{i}"]; F[f"lb{i}_m5"] = g(f"lb{i}").transform(lambda s: s.rolling(5, min_periods=1).mean())
    F[f"lb{i}_rise5"] = g(f"lb{i}").transform(lambda s: s.diff(5))
for i in range(1, len(BANDS)):
    F[f"hard_{i}"] = F[f"lb{i}_m5"] - F["lb0_m5"]
    F[f"hard_{i}_rise5"] = pd.Series(F[f"hard_{i}"].to_numpy(), index=df.index).groupby(df.date).diff(5)
X = pd.DataFrame(F); SPEC = [c for c in X if c not in BASE]
X[BASE] = X[BASE].fillna(0)                   # spectral features stay NaN where the PI spectrum is unavailable

days = sorted(df.date.unique()); test_days = set(days[-N_TEST_DAYS:])
tab = pd.concat([df[["date", "period"]].assign(split=np.where(df.date.isin(test_days), "test", "train"), has_data=df.valid),
                 df[["rate"]].rename(columns={"rate": "count_rate"}), df[["y_now", "y_forecast_30min"]].astype(int), X[BASE], X[SPEC],
                 df[[f"band{i}" for i in range(len(BANDS))]]], axis=1)
tab.index.name = "time_utc"; tab.to_csv(os.path.join(OUT, "training_table.csv"), float_format="%.6g")

# ---- 4. summary + data dictionary ----
print(f"\nrows {len(tab)} | columns {tab.shape[1]} | days {len(days)} | train rows {(tab.split == 'train').sum()} | test rows {(tab.split == 'test').sum()}")
print(f"flares {len(events)} | y_now positives {tab.y_now.sum()} ({tab.y_now.mean():.1%}) | y_forecast positives {tab.y_forecast_30min.sum()} ({tab.y_forecast_30min.mean():.1%})")
print(f"rows without data {int((~tab.has_data).sum())} | rows with spectral features {int(X[SPEC].notna().all(axis=1).sum())} | features: {len(BASE)} base + {len(SPEC)} spectral")
open(os.path.join(OUT, "DATA_DICTIONARY.md"), "w").write(f"""# training_table.csv

One row per minute, {len(tab):,} rows, {len(days)} days (7 event days in 2024, 11 in Mar 2025, 32 in 2026). Built by `build_dataset.py` from the SoLEXS SDD2 light curve and PI spectrum.

| Column | Meaning |
|---|---|
| `time_utc` | Start of the minute (UTC) |
| `date`, `period` | Day, as `events_2024`, `mar_2025` or `year_2026` |
| `split` | `train` (earlier days) or `test` (last {N_TEST_DAYS} days, 19-28 Oct 2026) |
| `has_data` | False where the minute has no valid seconds; exclude these rows from training |
| `count_rate` | Mean counts/s over the minute (empty if no data) |
| **`y_now`** | **Nowcast answer**: 1 if a flare is in progress (detector label) |
| **`y_forecast_30min`** | **Forecast answer**: 1 if a flare starts within the next 30 min (same day) |
| `{BASE[0]}` ... `{BASE[-1]}` | {len(BASE)} base features (past data only): log rate, rolling mean/max/std, background excess, rises, acceleration |
| `lb*`, `hard_*` | {len(SPEC)} spectral features: log band rates, 5-min means and rises, hardness vs the softest band (empty if no PI spectrum) |
| `band0` ... `band5` | Raw counts/s summed over six PI channel ranges {BANDS} (ordinal index, not keV) |

Labels come from a threshold detector on the same data (not an independent flare catalogue). Features never use future data.
""")
