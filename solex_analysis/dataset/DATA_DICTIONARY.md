# training_table.csv

One row per minute, 72,000 rows, 50 days (7 event days in 2024, 11 in Mar 2025, 32 in 2026). Built by `build_dataset.py` from the SoLEXS SDD2 light curve and PI spectrum.

| Column | Meaning |
|---|---|
| `time_utc` | Start of the minute (UTC) |
| `date`, `period` | Day, as `events_2024`, `mar_2025` or `year_2026` |
| `split` | `train` (earlier days) or `test` (last 10 days, 19-28 Oct 2026) |
| `has_data` | False where the minute has no valid seconds; exclude these rows from training |
| `count_rate` | Mean counts/s over the minute (empty if no data) |
| **`y_now`** | **Nowcast answer**: 1 if a flare is in progress (detector label) |
| **`y_forecast_30min`** | **Forecast answer**: 1 if a flare starts within the next 30 min (same day) |
| `mean_5` ... `accel_5` | 20 base features (past data only): log rate, rolling mean/max/std, background excess, rises, acceleration |
| `lb*`, `hard_*` | 28 spectral features: log band rates, 5-min means and rises, hardness vs the softest band (empty if no PI spectrum) |
| `band0` ... `band5` | Raw counts/s summed over six PI channel ranges [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 340)] (ordinal index, not keV) |

Labels come from a threshold detector on the same data (not an independent flare catalogue). Features never use future data.
