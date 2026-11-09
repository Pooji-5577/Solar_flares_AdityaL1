import sys, glob, gzip, io, zipfile, os
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Z, OUT = sys.argv[1], sys.argv[2]
def rd(zf, suf):
    n = [x for x in zf.namelist() if x.endswith(suf)][0]
    return fits.open(io.BytesIO(gzip.decompress(zf.read(n))))
fig, axs = plt.subplots(3, 1, figsize=(13, 10))
ts, cs = [], []
for z in sorted(glob.glob(os.path.join(Z, "*.zip"))):
    zf = zipfile.ZipFile(z); d = rd(zf, "SDD2_L1.lc.gz")[1].data
    ts.append(d["TIME"]); cs.append(d["COUNTS"])
t = np.concatenate(ts); c = np.concatenate(cs)
x = (t - t[0]) / 86400 + 0  # days since 2025-03-20
axs[0].plot(x, c, lw=.3); axs[0].set_yscale("log"); axs[0].set_title("SoLEXS SDD2 counts/s, 20-31 Mar 2025 (days since 20 Mar)")
zf = zipfile.ZipFile(os.path.join(Z, "AL1_SLX_L1_20250328_v1.0.zip"))
d = rd(zf, "SDD2_L1.lc.gz")[1].data; c28 = d["COUNTS"]; i = int(np.nanargmax(c28))
axs[1].plot(np.arange(86400)/3600, c28, lw=.4); axs[1].set_title(f"28 Mar 2025 (peak {c28[i]:.0f} cts/s at {i/3600:.2f} UT h)"); axs[1].set_xlabel("UT hour")
p = rd(zf, "SDD2_L1.pi.gz")[1].data
quiet = p["COUNTS"][:3600].sum(0); flare = p["COUNTS"][max(i-30,0):i+30].sum(0)
axs[2].step(np.arange(340), quiet/quiet.sum(), label="first hour"); axs[2].step(np.arange(340), flare/flare.sum(), label="60s around peak")
axs[2].set_yscale("log"); axs[2].set_xlabel("channel (340)"); axs[2].legend(); axs[2].set_title("Spectrum shape (normalised)")
plt.tight_layout(); plt.savefig(OUT, dpi=80)
print("exposure sample", p["EXPOSURE"][:3], "gaps(nan):", int(np.isnan(c).sum()))
