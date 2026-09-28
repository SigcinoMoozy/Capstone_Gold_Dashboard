"""
Feature / model configuration — extracted directly from the Capstone
notebook (5day_Movement.ipynb). This file is the single source of
truth the rest of the project imports from.

IMPORTANT — read this before changing anything
================================================
The notebook actually defines FOUR candidate feature lists (cell with
`SUS = [...]`), but only THREE are meaningful, and only ONE was ever
fed into a trained model:

    Tech        - pure technical/price-action features (39 cols)
    Macro       - the FULL fundamental/macro feature set (74 cols)
    Moozy_Tech  - a hand-curated HYBRID of technical + a trimmed macro
                  subset (54 cols) — THIS is what the notebook actually
                  did `Dataset = Dataset[Moozy_Tech]` with, and it is
                  the only feature set the XGBoost model in the
                  notebook was ever trained or evaluated on.

The brief that accompanied this project assumed a "Technical +
Fundamental" option would be `list(dict.fromkeys(Tech + Macro))` (a
naive deduplicated union, 105 columns). That combined list was
DEFINED in the notebook (as `Moozy`, a 45-ish/72-ish column list) but
never actually used for `Dataset[...]` and never trained on.

To stay faithful to "the notebook is the source of truth" instruction:
  - Technical              -> Tech
  - Fundamental             -> Macro
  - Technical + Fundamental -> Moozy_Tech (the notebook's real hybrid
                                combined set, NOT a naive Tech+Macro
                                union)

No saved .joblib model was uploaded alongside the notebook, so this
app trains fresh (per the notebook's exact methodology) rather than
loading a pre-trained model — see pipelines/training.py. The
dashboard also supports uploading a previously-saved .joblib produced
by THIS app (or by the notebook's own save cell), and will validate
its feature_cols against whatever feature group you select.
"""

# ══════════════════════════════════════════════════════════════
# FEATURE GROUPS  (verbatim from the notebook)
# ══════════════════════════════════════════════════════════════

TECH = [
    'date', 'open', 'high', 'low', 'close', 'tick_volume', 'spread',
    'real_volume', 'rsi_14', 'log_return_1d', 'log_return_10d',
    'log_return_21d', 'volatility_10d', 'dist_to_nearest_support',
    'dist_to_nearest_resistance', 'inside_zone',
    'currently_touching_zone', 'touching_support', 'touching_resistance',
    'nearest_support_strength', 'nearest_resistance_strength',
    'nearest_support_age', 'nearest_resistance_age',
    'support_tested_recently', 'resistance_tested_recently',
    'sr_compression', 'nearest_support_flipped',
    'nearest_resistance_flipped', 'ema_50', 'price_vs_ema',
    'ema_slope', 'trend_structure', 'pullback_detected',
    'pullback_depth', 'approaching_support',
    'approaching_resistance', 'approaching_zone',
    'trend_direction', 'trend_strength',
]

MACRO = [
    'date', 'open', 'high', 'low', 'close', 'tick_volume', 'spread',
    'real_volume', 'UNRATE_Unrate_release_date', 'UNRATE_actual',
    'UNRATE_previous', 'UNRATE_surprise', 'UNRATE_Surprise_Direction',
    'NFP_NFP_release_date', 'NFP_actual', 'NFP_previous',
    'NFP_Jobs_Added', 'NFP_Previous_FF', 'NFP_NFP_Surprise',
    'NFP_Surprise_Direction', 'CPI_CPI_release_date', 'CPI_actual',
    'CPI_previous', 'CPI_surprise', 'CPI_CPI_MoM', 'CPI_CPI_YoY',
    'CPI_CPI_MoM_Prev', 'CPI_CPI_YoY_Prev',
    'CPI_CPI_MoM_Surprise', 'CPI_CPI_YoY_Surprise',
    'CPI_CPI_MoM_Direction', 'CPI_CPI_YoY_Direction',
    'CORE_CPI_Core_CPI_release_date', 'CORE_CPI_actual',
    'CORE_CPI_previous', 'CORE_CPI_surprise',
    'CORE_CPI_Core_CPI_MoM', 'CORE_CPI_Core_CPI_YoY',
    'CORE_CPI_Core_CPI_MoM_Prev', 'CORE_CPI_Core_CPI_YoY_Prev',
    'CORE_CPI_Core_CPI_MoM_Surprise',
    'CORE_CPI_Core_CPI_YoY_Surprise',
    'CORE_CPI_Core_CPI_MoM_Direction',
    'CORE_CPI_Core_CPI_YoY_Direction',
    'PCE_PCE_release_date', 'PCE_actual', 'PCE_previous',
    'PCE_surprise', 'PCE_PCE_MoM', 'PCE_PCE_YoY',
    'PCE_PCE_MoM_Prev', 'PCE_PCE_YoY_Prev',
    'PCE_PCE_MoM_Surprise', 'PCE_PCE_YoY_Surprise',
    'PCE_PCE_MoM_Direction', 'PCE_PCE_YoY_Direction',
    'CORE_PCE_Core_PCE_release_date', 'CORE_PCE_actual',
    'CORE_PCE_previous', 'CORE_PCE_surprise',
    'CORE_PCE_Core_PCE_MoM', 'CORE_PCE_Core_PCE_YoY',
    'CORE_PCE_Core_PCE_MoM_Prev', 'CORE_PCE_Core_PCE_YoY_Prev',
    'CORE_PCE_Core_PCE_MoM_Surprise',
    'CORE_PCE_Core_PCE_YoY_Surprise',
    'CORE_PCE_Core_PCE_MoM_Direction',
    'CORE_PCE_Core_PCE_YoY_Direction',
    'days_since_NFP', 'days_since_UNRATE', 'days_since_CPI',
    'days_since_CORE_CPI', 'days_since_PCE', 'days_since_CORE_PCE',
]

# The notebook's actual "combined" set used for `Dataset[Moozy_Tech]`.
# A hand-picked hybrid: full technical block + a TRIMMED macro subset
# (release dates, MoM only [no YoY], no raw actual/previous levels for
# CPI/PCE families, but raw actual/previous kept for UNRATE). This is
# NOT the same as Tech ∪ Macro — it deliberately drops many Macro
# columns (surprise on CPI/PCE, YoY variants, raw actual/prev levels)
# that the notebook author judged redundant/noisy for the combined model.
COMBINED = [
    'date', 'open', 'high', 'low', 'close', 'volatility_10d',
    'volatility_3d', 'log_return_1d', 'log_return_10d', 'log_return_3d',
    'log_return_21d', 'ema_50', 'dist_to_nearest_support',
    'dist_to_nearest_resistance', 'touching_support', 'touching_resistance',
    'nearest_support_strength', 'nearest_resistance_strength',
    'nearest_support_age', 'nearest_resistance_age',
    'support_tested_recently', 'resistance_tested_recently',
    'nearest_support_flipped', 'nearest_resistance_flipped',
    'trend_structure', 'pullback_detected', 'pullback_depth',
    'approaching_support', 'approaching_resistance',
    'UNRATE_Unrate_release_date', 'UNRATE_actual', 'UNRATE_previous',
    'UNRATE_surprise', 'UNRATE_Surprise_Direction', 'NFP_NFP_release_date',
    'NFP_Jobs_Added', 'NFP_Previous_FF', 'NFP_NFP_Surprise',
    'NFP_Surprise_Direction', 'CPI_CPI_release_date', 'CPI_CPI_MoM',
    'CPI_CPI_MoM_Prev', 'CPI_CPI_MoM_Surprise', 'CPI_CPI_MoM_Direction',
    'CORE_CPI_Core_CPI_release_date', 'CORE_CPI_Core_CPI_MoM',
    'CORE_CPI_Core_CPI_MoM_Prev', 'CORE_CPI_Core_CPI_MoM_Surprise',
    'CORE_CPI_Core_CPI_MoM_Direction', 'PCE_PCE_release_date',
    'PCE_PCE_MoM', 'PCE_PCE_MoM_Prev', 'PCE_PCE_MoM_Surprise',
    'PCE_PCE_MoM_Direction', 'CORE_PCE_Core_PCE_release_date',
    'CORE_PCE_Core_PCE_MoM', 'CORE_PCE_Core_PCE_MoM_Prev',
    'CORE_PCE_Core_PCE_MoM_Surprise', 'CORE_PCE_Core_PCE_MoM_Direction',
    'days_since_NFP', 'days_since_UNRATE', 'days_since_CPI',
    'days_since_CORE_CPI', 'days_since_PCE', 'days_since_CORE_PCE',
]

FEATURE_GROUPS = {
    "Technical": TECH,
    "Fundamental": MACRO,
    "Technical + Fundamental": COMBINED,
}

# ══════════════════════════════════════════════════════════════
# TARGET LABELING  (label_weekly_first_touch_daily, notebook cell)
# ══════════════════════════════════════════════════════════════
LABEL_UP_THRESHOLD = 0.02      # +2% weekly first-touch -> BUY/LONG
LABEL_DOWN_THRESHOLD = -0.02   # -2% weekly first-touch -> SELL/SHORT
LABEL_MIN_DAYS_IN_WEEK = 4

# Columns produced by the labeling function that must NEVER be fed to
# the model — they encode the label / its timing directly.
DROP_COLS_WEEKLY = [
    'week_open', 'n_days', 'entry_date', 'exit_date',
    'trigger_date', 'trigger_day_offset',
]

# Columns that must always be excluded from the feature matrix
# regardless of feature group (raw OHLCV / identifiers / the target
# itself / anything the labeling step attaches).
BASE_DROP_COLS = [
    'date', 'time', 'open', 'high', 'low', 'close',
    'tick_volume', 'spread', 'real_volume', 'target',
    'log_return_5d',
    'UNRATE_Unrate_release_date', 'NFP_NFP_release_date',
    'CPI_CPI_release_date', 'CORE_CPI_Core_CPI_release_date',
    'PCE_PCE_release_date', 'CORE_PCE_Core_PCE_release_date',
] + DROP_COLS_WEEKLY

# ══════════════════════════════════════════════════════════════
# LABEL MAPPING  (notebook: -1/0/1 -> SHORT/NEUTRAL/LONG)
# ══════════════════════════════════════════════════════════════
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}
CLASSES = [-1, 0, 1]
CLASS_NAMES = ["SHORT (-1)", "NEUTRAL (0)", "LONG (1)"]
# Human-facing trading-signal names, same order as CLASSES
SIGNAL_NAMES = {-1: "SELL", 0: "HOLD", 1: "BUY"}

# ══════════════════════════════════════════════════════════════
# TRAINING CONFIG  (notebook's final "imbalance-aware" XGBoost cell —
# this supersedes the two earlier XGBoost cells in the notebook: it
# replaces a fixed EXTREME_WEIGHT_MULT with an automatic sweep, which
# is the last and most refined version defined before the model-save
# cell runs, so it is the version whose `xgb`/`feature_cols`/
# `thresholds` variables are what actually got saved to disk.)
# ══════════════════════════════════════════════════════════════
VAL_FRACTION = 0.15
WEIGHT_POWERS_TO_TRY = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
MIN_ACCEPTABLE_RECALL = 0.10

XGB_SWEEP_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    gamma=0.5, reg_lambda=2.0,
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    early_stopping_rounds=20, tree_method='hist', max_bin=256,
    nthread=2, random_state=42,
)

XGB_FINAL_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    gamma=0.5, reg_lambda=2.0,
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    early_stopping_rounds=30, tree_method='hist', max_bin=256,
    nthread=2, random_state=42,
)

THRESHOLD_GRID_START = 0.15
THRESHOLD_GRID_STOP = 0.60
THRESHOLD_GRID_STEP = 0.01

MODEL_DIR = "models"
MODEL_FILENAMES = {
    "Technical": "xgboost_technical.joblib",
    "Fundamental": "xgboost_fundamental.joblib",
    "Technical + Fundamental": "xgboost_combined.joblib",
}
