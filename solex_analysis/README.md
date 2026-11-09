# SoLEXS nowcast + forecast (Aditya-L1, 20-31 Mar 2025)

Data: `../solex_data/data/*.zip` (11 days of SoLEXS L1, SDD2). Method: AdityaNet protocol (`../data_analysis`).
No external data used. Flares are labelled from SoLEXS itself, so nowcast scores measure agreement with that detector.

| File | Purpose |
|---|---|
| `solex_common.py` | Load zips -> 1-min series; label flares (66 found) |
| `run_analysis.py` | First run: 11 base features, 3 models + baselines |
| `run_strong.py` | Stronger run: spectral features, leave-one-day-out tuning, ensemble |
| `export_dashboard.py` | Writes `../data_analysis/web/src/generated/data/solex/solex.json` for the /solex/ page |
| `results/` | JSON metrics, minute dataset, test scores, plots |
| `extras/` | Early exploration scripts and screenshots |

Run (from any folder, using the PBL venv):
    ../.venv/bin/python -I run_strong.py && ../.venv/bin/python -I export_dashboard.py

Dashboard page: `../data_analysis/web/src/pages/solex.astro`. Start it with
    cd ../data_analysis/web && npx pnpm@10 dev --port 4321   ->  http://127.0.0.1:4321/solex/

Results (test days 27-31 Mar): nowcast ROC-AUC ~0.98 (persistence 0.976); forecast ~0.50 (threshold 0.56) -- not learnable from 11 days.
