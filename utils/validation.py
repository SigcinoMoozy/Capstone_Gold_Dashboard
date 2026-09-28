"""
Human-readable validation for uploaded datasets and models. Every
function returns (ok: bool, messages: list[str]) rather than raising,
so the dashboard can show all problems at once instead of stopping at
the first one.
"""

from __future__ import annotations

import pandas as pd

from config import feature_config as cfg


def validate_raw_dataset(df: pd.DataFrame, label: str = "dataset") -> tuple[bool, list[str]]:
    msgs = []
    ok = True

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        ok = False
        msgs.append(f"[{label}] Missing required columns: {sorted(missing)}")

    if len(df) == 0:
        ok = False
        msgs.append(f"[{label}] File contains zero rows.")

    if "target" in df.columns:
        msgs.append(f"[{label}] Note: a 'target' column already exists in the upload — "
                     f"it will be RE-DERIVED by the weekly first-touch labeling step "
                     f"(the notebook's methodology), not read from the file.")

    if ok:
        n_dupe_dates = df["date"].duplicated().sum() if "date" in df.columns else 0
        if n_dupe_dates:
            msgs.append(f"[{label}] Warning: {n_dupe_dates} duplicate date rows found.")

    return ok, msgs


def validate_feature_group(df: pd.DataFrame, group_name: str, label: str = "dataset") -> tuple[bool, list[str], list[str]]:
    """
    Returns (ok, messages, resolved_feature_cols).
    """
    if group_name not in cfg.FEATURE_GROUPS:
        return False, [f"Unknown feature group '{group_name}'."], []

    wanted = cfg.FEATURE_GROUPS[group_name]
    available = set(df.columns)
    present = [c for c in wanted if c in available]
    missing = [c for c in wanted if c not in available]

    msgs = []
    ok = True
    if missing:
        # Some columns (date/OHLCV) overlap across groups and are expected;
        # only warn loudly if a large fraction of the group is missing.
        pct_missing = len(missing) / len(wanted) * 100
        severity = "ERROR" if pct_missing > 30 else "Warning"
        if severity == "ERROR":
            ok = False
        msgs.append(
            f"[{label}] {severity}: {len(missing)}/{len(wanted)} columns for "
            f"'{group_name}' are missing from the upload ({pct_missing:.0f}%). "
            f"Missing: {missing[:15]}{' ...' if len(missing) > 15 else ''}"
        )

    model_feature_cols = [c for c in present if c not in set(cfg.BASE_DROP_COLS)]
    if len(model_feature_cols) == 0:
        ok = False
        msgs.append(f"[{label}] No usable model features remain for '{group_name}' after "
                     f"excluding OHLCV/date/target/leakage columns.")

    return ok, msgs, model_feature_cols


def validate_model_compatibility(selected_feature_cols: list[str], model_feature_cols: list[str]) -> tuple[bool, str]:
    """
    Compare the feature columns the user selected/derived against what
    a LOADED model actually expects (model_package['feature_cols']).
    Order and exact membership both matter for a tree ensemble.
    """
    if selected_feature_cols == model_feature_cols:
        return True, f"✓ Compatible — {len(model_feature_cols)} features match exactly (name and order)."

    same_set_diff_order = (
        set(selected_feature_cols) == set(model_feature_cols)
        and selected_feature_cols != model_feature_cols
    )
    if same_set_diff_order:
        return True, (f"✓ Compatible — same {len(model_feature_cols)} features present; "
                       f"will be reordered to match the model's training order automatically.")

    missing_for_model = [c for c in model_feature_cols if c not in selected_feature_cols]
    extra_in_selection = [c for c in selected_feature_cols if c not in model_feature_cols]

    msg = (
        f"✗ Incompatible — model expects {len(model_feature_cols)} features, "
        f"selection provides {len(selected_feature_cols)}.\n"
        f"    Missing (needed by model, not in selection): {missing_for_model[:10]}"
        f"{' ...' if len(missing_for_model) > 10 else ''}\n"
        f"    Extra (in selection, unused by model): {extra_in_selection[:10]}"
        f"{' ...' if len(extra_in_selection) > 10 else ''}"
    )
    return False, msg


def validate_model_package(pkg: dict) -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    required_keys = ["model", "feature_cols"]
    for k in required_keys:
        if k not in pkg:
            ok = False
            msgs.append(f"Uploaded model file is missing required key '{k}'.")

    optional_keys = ["thresholds", "label_map", "label_map_inv", "classes", "class_names"]
    missing_optional = [k for k in optional_keys if k not in pkg]
    if missing_optional:
        msgs.append(
            f"Uploaded model is missing optional metadata {missing_optional} — "
            f"falling back to this app's default label mapping/thresholds. "
            f"If the model was trained differently, predictions may be mislabeled."
        )

    if ok and not hasattr(pkg["model"], "predict_proba"):
        ok = False
        msgs.append("Uploaded model object has no predict_proba() — expected a classifier.")

    return ok, msgs
