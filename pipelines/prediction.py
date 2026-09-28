"""
Prediction pipeline — turn a model package + a row of features into a
human-readable BUY/HOLD/SELL signal with probabilities, using the same
threshold-tuned decision rule used in evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import feature_config as cfg
from pipelines.preprocessing import downcast
from pipelines.training import predict_with_thresholds


def predict_rows(model_package: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    df must contain (at least) model_package['feature_cols']. Missing
    engineered columns raise a clear error rather than silently
    producing a garbage prediction.
    """
    model = model_package["model"]
    feature_cols = model_package["feature_cols"]
    thresholds = model_package["thresholds"]
    label_map_inv = model_package["label_map_inv"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot predict — {len(missing)} feature column(s) the model "
            f"expects are missing from the uploaded data: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    X = downcast(df[feature_cols].copy())
    proba = model.predict_proba(X)

    pred_argmax_enc = proba.argmax(axis=1)
    pred_tuned_enc = predict_with_thresholds(proba, thresholds)

    out = pd.DataFrame({
        "signal_argmax": [cfg.SIGNAL_NAMES[label_map_inv[p]] for p in pred_argmax_enc],
        "signal_tuned": [cfg.SIGNAL_NAMES[label_map_inv[p]] for p in pred_tuned_enc],
        "prob_SHORT": proba[:, 0],
        "prob_NEUTRAL": proba[:, 1],
        "prob_LONG": proba[:, 2],
        "confidence": proba.max(axis=1),
    })
    if "date" in df.columns:
        out.insert(0, "date", df["date"].values)
    return out


def predict_latest(model_package: dict, df: pd.DataFrame) -> dict:
    """Convenience wrapper: predict only the most recent row of df
    (by 'date' if present, else the last row) for the dashboard's
    single-prediction panel."""
    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date")
    row = d.tail(1)
    result = predict_rows(model_package, row)
    r = result.iloc[0]
    return {
        "date": r.get("date"),
        "signal": r["signal_tuned"],
        "signal_argmax": r["signal_argmax"],
        "probability": float(r["confidence"]),
        "prob_SHORT": float(r["prob_SHORT"]),
        "prob_NEUTRAL": float(r["prob_NEUTRAL"]),
        "prob_LONG": float(r["prob_LONG"]),
    }
