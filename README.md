# Gold Market Prediction System

A reproducible ML application built from the `5day_Movement.ipynb`
Capstone notebook: predicts weekly XAUUSD direction (SHORT / NEUTRAL /
LONG) from technical and fundamental (macro) daily features, with a
Streamlit dashboard for uploading data, selecting a feature group,
training/loading a model, and reviewing performance.

## 1. Project purpose

The notebook engineers daily technical features (RSI, EMA, trend
structure, an event-driven support/resistance zone tracker) and
fundamental/macro features (NFP, CPI, Core CPI, PCE, Core PCE, UNRATE
pulled from FRED, release-date aligned to avoid lookahead), labels a
weekly first-touch target (±2% from Monday's open), and trains an
XGBoost classifier. This project turns that notebook into a modular
application you can run and re-run without touching Jupyter.

**What this app does NOT re-implement**: the raw feature engineering
(MetaTrader5 pulls, FRED API pulls, the S/R zone tracker) requires a
live MT5 terminal and a FRED API key, and isn't portable to a
Streamlit app. This app picks up exactly where your own
`Dataset_Dashboard.csv` / `Dataset_Test_Dashboard.csv` files leave
off: an already feature-engineered daily dataset with no target
column yet. Upload files in that shape.

## 2. Project structure

```
Capstone_Project/
├── main.py                  # master script — CLI pipeline + launches dashboard
├── dashboard.py              # Streamlit app
├── requirements.txt
├── README.md
├── config/
│   └── feature_config.py    # feature groups, labeling params, XGBoost config
├── pipelines/
│   ├── preprocessing.py     # loading, weekly first-touch labeling, feature selection
│   ├── training.py          # weight-power sweep + final XGBoost + threshold tuning
│   ├── evaluation.py        # metrics, confusion matrix, feature importance
│   └── prediction.py        # turn a row of features into a BUY/HOLD/SELL signal
├── utils/
│   ├── validation.py        # dataset / feature-group / model-compatibility checks
│   └── helpers.py           # model save/load
├── data/
│   ├── training/             # put a training CSV here for the CLI pre-run
│   └── testing/               # put a testing CSV here for the CLI pre-run
├── models/                    # trained models land here (.joblib)
└── notebooks/
    └── capstone.ipynb        # your original notebook, for reference
```

## 3. Installation

```bash
pip install -r requirements.txt
```

(The original notebook additionally uses `MetaTrader5`, `fredapi`,
`ta`, `mplfinance`, `lightgbm`, `seaborn` for the raw feature
engineering and the LightGBM/Decision Tree/Random Forest comparison
cells. None of those are required to run this app, since it consumes
already-engineered CSVs and only serves the XGBoost model — the one
model the notebook actually saved.)

## 4. How to run

```bash
python main.py
```

This will:
1. Validate the project folders exist.
2. If exactly one CSV sits in `data/training/` and one in
   `data/testing/`, run the full pipeline once for the default
   feature group (`Technical + Fundamental`) — label, train (or load
   an existing saved model for that group), evaluate, and print a
   summary. This step is skipped if those folders don't contain
   exactly one CSV each; use the dashboard instead.
3. Launch the Streamlit dashboard (`dashboard.py`).

Useful flags:
```bash
python main.py --feature-group "Technical"      # CLI pre-run on a different group
python main.py --force-retrain                  # ignore any existing saved model
python main.py --no-dashboard                    # CLI pipeline only, skip Streamlit
```

Or run the dashboard directly:
```bash
streamlit run dashboard.py
```

## 5. How to upload datasets

The dashboard has two separate upload widgets in the sidebar —
**Training Dataset** and **Testing Dataset**. Both expect a CSV with
the same shape as `Dataset_Dashboard.csv` / `Dataset_Test_Dashboard.csv`:
a `date` column, raw OHLCV, and the full set of engineered technical +
fundamental columns. No `target` column is expected — it's derived
by the app.

## 6. How to upload the model

Optional. Under **Saved Model**, upload a `.joblib` file. It should
contain a dict with (at minimum) `model` and `feature_cols` keys —
the same shape this app's own save step produces, and the same shape
the notebook's own save cell produces:

```python
{
    "model": xgb,
    "feature_cols": feature_cols,
    "thresholds": thresholds,
    "label_map": LABEL_MAP,
    "label_map_inv": LABEL_MAP_INV,
    "classes": CLASSES,
    "class_names": CLASS_NAMES,
}
```

If `thresholds` / `label_map` / etc. are missing, the app falls back
to its own defaults and warns you — it never silently invents a
different label scheme.

## 7. How feature selection works

Choose one of three groups in the sidebar:

- **Technical** — pure price-action / S/R-zone / trend features.
- **Fundamental** — the full macro release set.
- **Technical + Fundamental** — the notebook's own curated hybrid
  (`Moozy_Tech`), **not** a naive union of the other two. See the
  "About the feature groups" note on the dashboard's main page for
  exactly which columns are included/excluded and why.

Selecting a group resolves it against whatever columns are actually
present in your uploaded data, then removes anything that must never
reach the model (raw OHLCV, `date`, `target`, and the labeling
function's own bookkeeping columns like `trigger_day_offset` — these
would leak the label into the features).

## 8. How model compatibility is enforced

If you choose **"Use uploaded saved model"**, the app compares the
feature columns your selection resolves to against the model's own
`feature_cols`:

- Exact match (name + order) → compatible, runs as-is.
- Same features, different order → compatible, reordered automatically.
- Mismatched feature sets → **blocked**, with a clear list of which
  columns the model needs that your selection doesn't have, and which
  extra columns your selection has that the model won't use. The app
  will not silently subset or pad a mismatched feature matrix.

If you choose **"Train fresh"**, this check doesn't apply — a new
model is trained on exactly the feature group you selected, and saved
to `models/xgboost_<group>.joblib` (technical / fundamental /
combined) so different feature groups don't overwrite each other.

## 9. How evaluation works

The training set is used entirely for training, with the **last 15%
sliced off chronologically** (never shuffled — this is time-series
data) as an internal validation slice used only for early stopping,
the sample-weight sweep, and SHORT/LONG threshold tuning. The uploaded
testing dataset is never touched during training and is the only
source for the metrics shown: accuracy, macro ROC-AUC, log loss, a
full classification report, PR-AUC per class, and a confusion matrix.
Metrics are shown for both the naive argmax prediction and the
threshold-tuned prediction (tuned is the notebook's intended
production rule — argmax alone tends to collapse everything to
NEUTRAL under this dataset's class imbalance).

## 10. How to interpret the output

- **Signal**: SELL (SHORT), HOLD (NEUTRAL), or BUY (LONG) for the
  most recent row in your test set.
- **Probability**: the model's confidence in whichever class it chose.
- **Feature importance**: XGBoost's built-in gain-based importance,
  in the model's own training feature order.
- **Performance Comparison** (section 6 of the dashboard) only shows
  feature groups you've actually run in the current session — nothing
  is fabricated for groups you haven't tried.

## 11. How to add another model

The training/evaluation/prediction pipelines are written against a
plain `predict_proba`-supporting classifier and a small metadata dict
— they don't assume XGBoost specifically anywhere except
`pipelines/training.py`. To add e.g. LightGBM:

1. Add a `train_lightgbm()` function to `pipelines/training.py` that
   returns the same dict shape (`model`, `feature_cols`, `thresholds`,
   `label_map`, `label_map_inv`, `classes`, `class_names`).
2. Add a model-type selector in `dashboard.py`'s sidebar that calls
   the new function instead of `tr.train_xgboost`.
3. Everything downstream (evaluation, prediction, saving/loading,
   compatibility checks) works unchanged.

## Known uncertainties (flagged rather than guessed)

- **No trained `.joblib` was supplied** alongside the notebook — only
  the notebook and the two dashboard CSVs. This app trains fresh
  using the notebook's own final ("imbalance-aware weight sweep")
  XGBoost methodology rather than loading a pre-existing model. If you
  have a saved model file from your own run of the notebook, upload it
  under "Saved Model" instead.
- The notebook defines **four** feature-list variables in one cell
  (`SUS`, `Macro`, `Tech`, `Moozy`, `Moozy_Tech`); `SUS` and `Moozy`
  are dead code — never applied to `Dataset`. Only `Moozy_Tech` was
  ever actually used for training, which is why "Technical +
  Fundamental" here means `Moozy_Tech`, not `Tech ∪ Macro`.
- The notebook contains three successive XGBoost training cells
  (naive weights → fixed `EXTREME_WEIGHT_MULT` → automatic weight-power
  sweep). The last one is treated as authoritative here, since it's
  the version whose variables were in scope when the notebook's save
  cell ran.
