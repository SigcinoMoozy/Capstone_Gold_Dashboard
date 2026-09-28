"""
Model save/load helpers. Mirrors the notebook's own save cell:

    xgb_package = {
        "model": xgb,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "label_map": LABEL_MAP,
        "label_map_inv": LABEL_MAP_INV,
        "classes": CLASSES,
        "class_names": CLASS_NAMES,
    }
    joblib.dump(xgb_package, "models/xgboost_gold_classifier.joblib")

Note the notebook uses lowercase keys ("label_map", not "LABEL_MAP").
This app's defaults follow the same lowercase convention so a model
saved by this app or by the notebook loads identically. If you upload
a .joblib saved with different key casing, utils.validation will
still accept it as long as "model" and "feature_cols" are present,
and will fall back to this app's defaults for anything else missing.
"""

from __future__ import annotations

import os
import joblib

from config import feature_config as cfg


def model_path_for_group(group_name: str) -> str:
    filename = cfg.MODEL_FILENAMES.get(group_name)
    if filename is None:
        raise ValueError(f"No default filename configured for feature group '{group_name}'")
    return os.path.join(cfg.MODEL_DIR, filename)


def save_model_package(model_package: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model_package, path)
    return os.path.abspath(path)


def load_model_package(path_or_buffer) -> dict:
    pkg = joblib.load(path_or_buffer)
    if not isinstance(pkg, dict):
        raise ValueError(
            "Uploaded .joblib does not contain the expected dict package "
            "(model/feature_cols/thresholds/...). Got a bare "
            f"{type(pkg).__name__} instead."
        )
    # Fill in defaults for any missing optional metadata.
    pkg.setdefault("thresholds", {0: 0.33, 2: 0.33})
    pkg.setdefault("label_map", cfg.LABEL_MAP)
    pkg.setdefault("label_map_inv", cfg.LABEL_MAP_INV)
    pkg.setdefault("classes", cfg.CLASSES)
    pkg.setdefault("class_names", cfg.CLASS_NAMES)
    return pkg


def list_saved_models() -> list[str]:
    if not os.path.isdir(cfg.MODEL_DIR):
        return []
    return [
        os.path.join(cfg.MODEL_DIR, f)
        for f in sorted(os.listdir(cfg.MODEL_DIR))
        if f.endswith(".joblib")
    ]
