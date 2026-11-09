import sys, glob, gzip, io, zipfile, os
import numpy as np
from astropy.io import fits
Z = sys.argv[1]
def open_fits(zf, name):
    return fits.open(io.BytesIO(gzip.decompress(zf.read(name))))
first = True
tot = []
for z in sorted(glob.glob(os.path.join(Z, "*.zip"))):
    zf = zipfile.ZipFile(z)
    names = zf.namelist()
    lc = [n for n in names if n.endswith("SDD2_L1.lc.gz")][0]
    h = open_fits(zf, lc)
    if first:
        h.info()
        for i, hd in enumerate(h):
            if i < 2: print(repr(hd.header)[:2500] if i else "")
        print(h[1].columns)
        for kind in ("pi", "gti"):
            n = [x for x in names if x.endswith(f"SDD2_L1.{kind}.gz")][0]
            g = open_fits(zf, n); print("\n==", kind); g.info(); print(g[1].columns)
        n = [x for x in names if x.endswith("SDD1_L1.gti.gz")][0]
        g = open_fits(zf, n); print("SDD1 gti rows:", len(g[1].data))
        first = False
    d = h[1].data
    t = d["TIME"]; c = d["COUNTS"].astype(float)
    tot.append((os.path.basename(z)[11:19], len(t), t.min(), t.max(), np.nanmedian(c), np.nanmean(c), np.nanmax(c), np.isnan(c).sum()))
print("\ndate rows tmin tmax median mean max nan")
for r in tot: print(*r)
