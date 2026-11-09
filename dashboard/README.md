# Solar flare nowcast + forecast dashboard

A local web app (not the AdityaNet site) in the same visual design. Upload a one-minute feature table and the trained models
score every minute: nowcast (flare in progress now) and forecast (flare onset in the next 30 min).

Run (from `PBL/`):

    .venv/bin/python dashboard/server.py      # then open http://127.0.0.1:8765

- **Input:** a CSV in `solex_analysis/dataset/training_table.csv` format (time in the first column + the 20 base and 28 spectral
  feature columns; see `solex_analysis/dataset/DATA_DICTIONARY.md`). `y_now` / `y_forecast_30min` are optional and switch on AUC, TSS,
  flares caught and false-alert rates. `has_data` is optional.
- **Models:** `models/` holds the final models (trained on 33 SoLEXS days: 11 from Mar 2025, 22 from 2026; the other 10 days of the 43-day table were held out for testing). Ensemble = rank-average of logistic regression, random
  forest and gradient boosting.
- **Samples:** `samples/` (also the two "Try" buttons). Deep links: `/?sample=quiet`, `/?sample=flares`, add `&day=2024-02-22`.
- **Controls:** pick the model, a single day, and drag the alert thresholds; hover the charts for values.
- **Honest limits:** nowcast ~0.99 AUC but persistence is nearly as good; forecast ~0.6 AUC with ~80% false alerts. Labels come from a
  threshold detector, not an official flare list. Rows from the 33 training days were seen by the models.
