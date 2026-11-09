"""Load SoLEXS SDD2 light curves from the zips and derive a 1-min series + flare events (no external data)."""
import glob, gzip, io, os, zipfile
import numpy as np
import pandas as pd
from astropy.io import fits

DATA = os.path.join(os.path.dirname(__file__), "..", "solex_data", "data")

def load_minutes():
    """1-min mean count rate (cts/s) for every day; minutes with no valid seconds are NaN."""
    frames = []
    for z in sorted(glob.glob(os.path.join(DATA, "*.zip"))):
        zf = zipfile.ZipFile(z)
        name = [n for n in zf.namelist() if n.endswith("SDD2_L1.lc.gz")][0]
        d = fits.open(io.BytesIO(gzip.decompress(zf.read(name))))[1].data
        s = pd.Series(np.asarray(d["COUNTS"], float),
                      index=pd.to_datetime(np.asarray(d["TIME"], float), unit="s", utc=True))
        frames.append(s.resample("1min").mean())
    ts = pd.concat(frames).sort_index()
    return ts.rename("rate").to_frame()

def label_events(df, k=4.0, min_len=3, merge_gap=10):
    """Flares from a NON-causal (centred) background + robust sigma. Used only as ground truth."""
    lr = np.log1p(df.rate)
    bg = lr.rolling(181, center=True, min_periods=30).quantile(0.25)
    res = lr - bg
    sig = 1.4826 * (res - res.median()).abs().median()
    hot = (res.rolling(3, center=True, min_periods=1).median() > k * sig).fillna(False).to_numpy()
    ev, i, n = [], 0, len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]: j += 1
            ev.append([i, j]); i = j + 1
        else: i += 1
    merged = []
    for a, b in ev:
        if merged and a - merged[-1][1] <= merge_gap: merged[-1][1] = b
        else: merged.append([a, b])
    merged = [(a, b) for a, b in merged if b - a + 1 >= min_len]
    flare = np.zeros(n, bool)
    for a, b in merged: flare[a:b + 1] = True
    return flare, merged, float(sig)
