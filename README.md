# PBL: Solar Flare Detection and Forecasting Using SoLEXS Data and Machine Learning

Plan: `PBL_POOJITHA_1332240370.pdf` (page 2). Uses **Aditya-L1 SoLEXS Level-1 data only** (no GOES, no external flare catalogue).
Presentation: `Solar_Flares_PBL.pptx` (built on the 11-day March 2025 results; see "Status of the deck").

## Data (`solex_data/`, 50 daily SoLEXS zips)
| Folder | Days | Notes |
|---|---|---|
| `data/` | 11 | 20-31 Mar 2025 (21 Mar missing) |
| `data_2026/` | 32 | Feb, Apr, Jun and 26 days of Oct 2026 |
| `flare_event_data/` | 7 | Big-flare days: 22-23 Feb 2024 (X6.3 flare peaked 22:32 UTC), 7-11 May 2024 |

Each zip has SDD2 `.lc.gz` (1 s light curve), `.pi.gz` (340-channel spectrum, 1 s) and `.gti.gz`; SDD1 gives a GTI file only.
Files are gzip FITS read with Astropy. PI channels are an ordinal index; the archive has no keV calibration.

## Pipeline (`solex_analysis/`)
| Step | File | Output |
|---|---|---|
| 1. FITS to table | `build_dataset.py` | `dataset/training_table.csv`: 72,000 rows (one per minute), 20 base + 28 spectral features, labels `y_now` and `y_forecast_30min`, train/test split. Columns explained in `dataset/DATA_DICTIONARY.md` |
| 2. Train both tasks | `train_models.py [auto\|base]` | Nowcast and forecast models (`models_*.joblib`), metrics, test scores |
| 3. Out-of-regime test | `evaluate_event_days.py` | Final models scored on the 7 event days |
| Earlier scripts | `solex_common.py`, `run_analysis.py`, `run_strong.py`, `run_forecast_2026.py`, `run_forecast_combined.py` | First experiments (11-day March 2025 run behind the deck) |

Run from `PBL/`: `.venv/bin/python -I solex_analysis/build_dataset.py`, then `train_models.py`, then `evaluate_event_days.py`.

## Method
- **Labels** come from a 4σ threshold detector on SoLEXS itself (each period labelled separately): 295 flares in 50 days. Not an independent flare catalogue.
- **Tasks:** nowcast = flare in progress now; forecast = flare onset in the next 30 min.
- **Features** use past data only, computed per day. Models: logistic regression, random forest, gradient boosting (class-balanced) and a rank-average ensemble; 5 time-ordered folds choose hyperparameters and base vs base+spectral features.
- **Test:** the last 10 days (19-28 Oct 2026, 69 flares), scored once; threshold picked on training data.

## Final models: `solex_analysis/results_table_43days/`
Trained on 33 days (11 from Mar 2025 + 22 from 2026, up to 18 Oct); the last 10 days (19-28 Oct 2026) are the test set and the 7 event days are held out. Test AUC (0.5 = chance):

| Model | Nowcast | Forecast |
|---|---|---|
| Logistic | 0.987 | 0.567 |
| Random forest | 0.981 | **0.615** [0.58-0.67] |
| Gradient boosting | 0.987 | 0.599 |
| Ensemble | **0.988** | 0.603 |
| Threshold rule | 0.951 | 0.547 |
| Persistence | 0.968 | 0.467 |

On the 7 unseen event days (`results_event_days/`): nowcast ensemble 0.976 (persistence 0.981); forecast random forest 0.657 [0.61-0.71], ensemble 0.648, threshold rule 0.442.

Training on all 50 days (`results_table/`) did **not** help: forecast AUC fell to 0.58 (nowcast stayed 0.988). The event days are far more active (median 80-899 counts/s vs 7-33), so mixing them in shifts the background the models learn.

## Conclusions and caveats
- **Nowcasting works** (AUC about 0.98) but persistence is almost as good; labels are detector-made, so this is agreement with the detector.
- **Forecasting is weak**: AUC 0.6-0.66, better than chance and than the threshold rule, but about 80% of alert runs are false alarms (TSS 0.1-0.23). Not operationally useful yet.
- More data helped (11 days: chance; 33 training days: 0.6). A real flare list (e.g. NOAA) for labels and many more quiet-period days are the next steps.

## Deck and dashboard
- `Solar_Flares_PBL.pptx`: 12 slides with the final results (50 days of data, 33-day training set, test days, big-flare-day test, 17-day accuracy summary, dashboard). The longer 20-slide version is kept as `Solar_Flares_PBL_full20.pptx`.
- `dashboard/`: local web app to upload a training-table CSV and see nowcast/forecast scores (`.venv/bin/python dashboard/server.py`, then open http://127.0.0.1:8765). Ten extra test CSVs and per-file accuracy are in `dashboard/samples/more_tests/`.
