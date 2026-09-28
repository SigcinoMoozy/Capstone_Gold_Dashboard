"""
Evaluation pipeline — metrics ported from the notebook's evaluate()
function, computed for BOTH the naive argmax prediction and the
threshold-tuned prediction, on a fully separate test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    log_loss,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

from config import feature_config as cfg
from pipelines.preprocessing import prepare_xy
from pipelines.training import predict_with_thresholds


def _evaluate_one(y_true, y_pred, y_prob, y_bin) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=cfg.CLASS_NAMES, digits=4, output_dict=True, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=cfg.CLASSES)

    pr_auc = {}
    for idx, name in enumerate(cfg.CLASS_NAMES):
        ap = average_precision_score(y_bin[:, idx], y_prob[:, idx])
        prevalence = y_bin[:, idx].mean()
        pr_auc[name] = {
            "pr_auc": ap,
            "prevalence": prevalence,
            "lift": (ap / prevalence) if prevalence > 0 else float("nan"),
        }

    return {
        "accuracy": acc,
        "roc_auc_macro": auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "pr_auc_per_class": pr_auc,
    }


def evaluate_model(model_package: dict, df_test: pd.DataFrame) -> dict:
    """
    model_package: dict as returned by pipelines.training.train_xgboost
                   (or loaded via utils.helpers.load_model_package).
    df_test: labeled test dataframe (has a 'target' column).

    Returns metrics for both "argmax" and "tuned" prediction, plus the
    raw arrays needed for a confusion-matrix plot / feature importance.
    """
    model = model_package["model"]
    feature_cols = model_package["feature_cols"]
    thresholds = model_package["thresholds"]
    label_map = model_package["label_map"]
    label_map_inv = model_package["label_map_inv"]

    X_test, y_test, rows_before, rows_after = prepare_xy(df_test, feature_cols)
    if len(X_test) == 0:
        raise ValueError("No test rows remain after dropping missing targets — cannot evaluate.")

    y_test_enc = y_test.map(label_map).astype("int32")
    y_prob = model.predict_proba(X_test)

    y_pred_argmax_enc = y_prob.argmax(axis=1)
    y_pred_argmax = np.array([label_map_inv[p] for p in y_pred_argmax_enc])

    y_pred_tuned_enc = predict_with_thresholds(y_prob, thresholds)
    y_pred_tuned = np.array([label_map_inv[p] for p in y_pred_tuned_enc])

    y_true = y_test.values
    y_bin = label_binarize(y_true, classes=cfg.CLASSES)

    argmax_metrics = _evaluate_one(y_true, y_pred_argmax, y_prob, y_bin)
    tuned_metrics = _evaluate_one(y_true, y_pred_tuned, y_prob, y_bin)

    ll = log_loss(y_test_enc, y_prob)

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = (
            pd.Series(model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False)
        )

    return {
        "n_test_rows": len(X_test),
        "rows_before_target_drop": rows_before,
        "argmax": argmax_metrics,
        "tuned": tuned_metrics,
        "log_loss": ll,
        "feature_importance": importance,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred_argmax": y_pred_argmax,
        "y_pred_tuned": y_pred_tuned,
    }
