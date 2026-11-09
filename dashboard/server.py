"""Solar flare nowcast + forecast dashboard (local web app).

Run:  .venv/bin/python dashboard/server.py        then open http://127.0.0.1:8765
Upload a training_table.csv-format file (see solex_analysis/dataset/DATA_DICTIONARY.md); the saved final models
(trained on 33 SoLEXS days) score every minute. Labels (y_now, y_forecast_30min) are optional: with them the page
also shows AUC, TSS, flares caught and false alerts.
"""
import io, json, os, sys, warnings
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); STATIC = os.path.join(HERE, "static"); SAMPLES = os.path.join(HERE, "samples")
PORT = int(os.environ.get("PORT", "8765")); MAX_BYTES = 150 * 1024 * 1024; MAX_ROWS = 200_000
BUNDLES = {t: joblib.load(os.path.join(HERE, "models", f"models_{t}.joblib")) for t in ("nowcast", "forecast")}
REQUIRED = sorted({c for b in BUNDLES.values() for m in b["members"].values() for c in m["features"]})
NAMES = {"logistic": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting", "ensemble": "Ensemble"}
SAMPLE_FILES = {"quiet": "quiet_test_days_oct2026.csv", "flares": "big_flare_days_2024.csv"}
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml"}

def rank01(p, ref): return np.searchsorted(ref, p) / len(ref)

def predict(text):
    df = pd.read_csv(io.StringIO(text), index_col=0, low_memory=False)
    if len(df) == 0: raise ValueError("The file has no rows.")
    if len(df) > MAX_ROWS: raise ValueError(f"Too many rows ({len(df):,}); the limit is {MAX_ROWS:,}.")
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    if idx.isna().all(): raise ValueError("The first column must be the time (UTC), e.g. 2025-03-20 00:00:00+00:00.")
    df = df[~idx.isna()]; idx = idx[~idx.isna()]; df.index = idx; df = df.sort_index()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} model input columns are missing, e.g. {', '.join(missing[:6])}. "
                         "Upload a file in training_table.csv format (see DATA_DICTIONARY.md).")
    valid = df["has_data"].astype(bool).to_numpy() if "has_data" in df.columns else np.ones(len(df), bool)
    n = len(df); out = {"now": {}, "fc": {}, "thr": {"now": {}, "fc": {}}, "auc": {"now": {}, "fc": {}}}
    labels = {"now": "y_now", "fc": "y_forecast_30min"}
    for key, task in (("now", "nowcast"), ("fc", "forecast")):
        b = BUNDLES[task]; probs = {}
        for name, m in b["members"].items():
            probs[name] = m["model"].predict_proba(df[m["features"]].fillna(0).to_numpy(float))[:, 1]
            out["thr"][key][name] = float(m["threshold"])
        probs["ensemble"] = np.mean([rank01(probs[nm], b["members"][nm]["oof_ref"]) for nm in b["members"]], axis=0)
        out["thr"][key]["ensemble"] = float(b["ensemble_threshold"])
        for name, p in probs.items(): out[key][name] = [round(float(v), 4) for v in p]
        if labels[key] in df.columns:
            y = df[labels[key]].astype(int).to_numpy()
            for name, p in probs.items():
                ok = valid & ~np.isnan(p)
                out["auc"][key][name] = float(roc_auc_score(y[ok], p[ok])) if 0 < y[ok].sum() < ok.sum() else None
    has_labels = "y_now" in df.columns and "y_forecast_30min" in df.columns
    t = ((df.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy().astype(np.int64)
    rate = df["count_rate"].to_numpy(float) if "count_rate" in df.columns else np.expm1(df["log_rate"].to_numpy(float)) if "log_rate" in df.columns else np.full(n, np.nan)
    if has_labels:
        y_now = df["y_now"].astype(int).to_numpy(); y_fc = df["y_forecast_30min"].astype(int).to_numpy()
        start = (y_now == 1) & (np.r_[0, y_now[:-1]] == 0) | ((y_now == 1) & np.r_[True, np.diff(t) > 60])
        run = np.cumsum(start) - 1; ev_id = np.where(y_now == 1, run, -1); starts = np.where(start)[0]
        nxt = np.searchsorted(starts, np.arange(n), side="right"); on_id = np.where((y_fc == 1) & (nxt < len(starts)), nxt, -1)
        out.update({"y_now": y_now.tolist(), "y_fc": y_fc.tolist(), "ev_id": ev_id.tolist(), "on_id": on_id.tolist()})
    seen = None
    if "period" in df.columns and "split" in df.columns:        # rows the final models trained on: the 33 training days (not the 10 test days, not the event days)
        seen = int(((df["split"] == "train") & (df["period"] != "events_2024") & valid).sum())
    days = sorted({d for d in df.index.strftime("%Y-%m-%d")})
    out.update({"t": t.tolist(), "rate": [None if np.isnan(v) else round(float(v), 2) for v in rate], "valid": valid.astype(int).tolist(),
                "labels": bool(has_labels), "days": days, "n": int(n), "seen_rows": seen, "names": NAMES,
                "models_note": "Final models: trained on 33 SoLEXS days (Mar 2025 + early 2026)."})
    return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def _json(self, code, obj): self._send(code, json.dumps(obj, separators=(",", ":")))
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/sample":
            name = parse_qs(u.query).get("name", [""])[0]
            if name not in SAMPLE_FILES: return self._json(404, {"error": "Unknown sample."})
            try:
                with open(os.path.join(SAMPLES, SAMPLE_FILES[name]), encoding="utf-8") as f: return self._json(200, predict(f.read()))
            except Exception as e: return self._json(400, {"error": str(e)})
        path = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
        if path.startswith("static/"): path = path[len("static/"):]
        full = os.path.normpath(os.path.join(STATIC, path))
        if not full.startswith(STATIC) or not os.path.isfile(full): return self._send(404, "Not found", "text/plain")
        with open(full, "rb") as f: self._send(200, f.read(), CTYPES.get(os.path.splitext(full)[1], "application/octet-stream"))
    def do_POST(self):
        if urlparse(self.path).path != "/api/predict": return self._json(404, {"error": "Not found."})
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > MAX_BYTES: return self._json(400, {"error": "Empty or too large file (limit 150 MB)."})
        try: return self._json(200, predict(self.rfile.read(n).decode("utf-8-sig")))
        except Exception as e: return self._json(400, {"error": str(e)})

if __name__ == "__main__":
    print(f"Solar flare dashboard: http://127.0.0.1:{PORT}  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
