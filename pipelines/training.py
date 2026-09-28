"""
Training pipeline — ported from the notebook's final, "imbalance-aware"
XGBoost cell. This is the version whose `xgb` / `feature_cols` /
`thresholds` variables are the ones the notebook's own save-cell
persists (it's the last cell before saving that assigns those names),
so it is treated as the authoritative training methodology here.

Sequence (matches the notebook exactly):
  1. Drop rows with missing target.
  2. Chronological internal train/val split (last VAL_FRACTION of the
     training set) — used ONLY for the weight sweep, early stopping,
     and SHORT/LONG threshold tuning. The held-out test set is never
     touched here.
  3. Sweep sample-weight "damping" powers on the val slice, reject any
     power that crushes a class's recall below MIN_ACCEPTABLE_RECALL,
     pick the surviving power with the best macro-F1.
  4. Refit final XGBoost with the selected weighting.
  5. Tune SHORT/LONG probability thresholds on the val slice (argmax
     alone tends to collapse everything to NEUTRAL under heavy
     imbalance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score

from config import feature_config as cfg
from pipelines.preprocessing import prepare_xy


def build_sample_weights(y_enc: pd.Series, power: float = 1.0):
    """
    power=1.0 -> full inverse-frequency (can crush the majority class)
    power=0.0 -> no reweighting (majority class dominates)
    0 < power < 1 -> damped inverse-frequency
    """
    counts = y_enc.value_counts()
    total = len(y_enc)
    inv_freq = {c: total / (len(counts) * cnt) for c, cnt in counts.items()}
    base_w = {c: w ** power for c, w in inv_freq.items()}
    return y_enc.map(base_w).values.astype("float32"), base_w


def tune_thresholds(y_val_true: np.ndarray, proba_val: np.ndarray) -> dict:
    """Sweep a probability cutoff for SHORT (col 0) and LONG (col 2),
    maximizing one-vs-rest F1 for each on the validation slice."""
    thresholds = {}
    grid = np.arange(cfg.THRESHOLD_GRID_START, cfg.THRESHOLD_GRID_STOP, cfg.THRESHOLD_GRID_STEP)
    for cls_enc, cls_col in [(0, 0), (2, 2)]:
        y_bin = (y_val_true == cls_enc).astype(int)
        best_f1, best_t = -1, 0.33
        for t in grid:
            pred_bin = (proba_val[:, cls_col] >= t).astype(int)
            if pred_bin.sum() == 0:
                continue
            f1 = f1_score(y_bin, pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[cls_enc] = float(best_t)
    return thresholds


def predict_with_thresholds(proba: np.ndarray, thresholds: dict) -> np.ndarray:
    """Assign SHORT(0)/LONG(2) if their tuned threshold clears, else
    fall back to argmax over {SHORT, NEUTRAL, LONG}."""
    n = proba.shape[0]
    preds = np.full(n, 1, dtype=int)

    short_ok = proba[:, 0] >= thresholds[0]
    long_ok = proba[:, 2] >= thresholds[2]

    argmax_default = proba.argmax(axis=1)
    preds[~short_ok & ~long_ok] = argmax_default[~short_ok & ~long_ok]

    only_short = short_ok & ~long_ok
    only_long = long_ok & ~short_ok
    both = short_ok & long_ok

    preds[only_short] = 0
    preds[only_long] = 2
    preds[both] = np.where(proba[both, 0] >= proba[both, 2], 0, 2)

    return preds


def train_xgboost(df_train: pd.DataFrame, feature_cols: list[str], log=print) -> dict:
    """
    Full training sequence on df_train (already has a 'target' column
    from preprocessing.label_weekly_first_touch_daily). Returns a dict
    with everything needed for evaluation/prediction/saving.
    """
    X_train, y_train, rows_before, rows_after = prepare_xy(df_train, feature_cols)
    log(f"Rows before target NaN drop: {rows_before} | after: {rows_after}")

    y_train_enc = y_train.map(cfg.LABEL_MAP).astype("int32")

    val_split = int(len(X_train) * (1 - cfg.VAL_FRACTION))
    if val_split < 10 or (len(X_train) - val_split) < 10:
        raise ValueError(
            f"Not enough rows ({len(X_train)}) to make a meaningful "
            f"chronological train/val split. Upload more training data."
        )

    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train_enc.iloc[:val_split], y_train_enc.iloc[val_split:]

    log(f"Internal train slice: {len(X_tr)} rows | val slice: {len(X_val)} rows")

    # ── STEP: weight-power sweep ─────────────────────────────
    sweep_results = []
    for power in cfg.WEIGHT_POWERS_TO_TRY:
        sw_tr_p, weight_map_p = build_sample_weights(y_tr, power=power)
        m = XGBClassifier(**cfg.XGB_SWEEP_PARAMS)
        m.fit(X_tr, y_tr, sample_weight=sw_tr_p, eval_set=[(X_val, y_val)], verbose=False)

        val_pred = m.predict(X_val)
        macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        per_class_rec = recall_score(y_val, val_pred, average=None, labels=[0, 1, 2], zero_division=0)

        sweep_results.append({
            "power": power, "macro_f1": macro_f1,
            "short_rec": per_class_rec[0], "neutral_rec": per_class_rec[1],
            "long_rec": per_class_rec[2], "weight_map": weight_map_p,
        })

    viable = [r for r in sweep_results
              if min(r["short_rec"], r["neutral_rec"], r["long_rec"]) >= cfg.MIN_ACCEPTABLE_RECALL]
    best = max(viable, key=lambda r: r["macro_f1"]) if viable else max(sweep_results, key=lambda r: r["macro_f1"])
    best_power = best["power"]

    log(f"Weight-power sweep -> selected power={best_power} "
        f"(macro-F1={best['macro_f1']:.4f}, recalls: "
        f"SHORT={best['short_rec']:.3f} NEUTRAL={best['neutral_rec']:.3f} LONG={best['long_rec']:.3f})")
    if not viable:
        log("WARNING: no sweep candidate kept every class recall above "
            f"{cfg.MIN_ACCEPTABLE_RECALL} — falling back to best macro-F1 overall.")

    # ── STEP: final fit with selected weighting ──────────────
    sw_tr, weight_map = build_sample_weights(y_tr, power=best_power)
    model = XGBClassifier(**cfg.XGB_FINAL_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=False)

    log(f"Final XGBoost — best iteration: {model.best_iteration}, "
        f"best val mlogloss: {model.best_score:.4f}")

    # ── STEP: tune SHORT/LONG thresholds on val slice ────────
    proba_val = model.predict_proba(X_val)
    thresholds = tune_thresholds(y_val.values, proba_val)
    log(f"Tuned thresholds — SHORT >= {thresholds[0]:.2f} | LONG >= {thresholds[2]:.2f}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "label_map": cfg.LABEL_MAP,
        "label_map_inv": cfg.LABEL_MAP_INV,
        "classes": cfg.CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "weight_power": best_power,
        "sweep_results": sweep_results,
        "n_train_rows": len(X_train),
    }
