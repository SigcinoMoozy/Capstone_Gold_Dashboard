"""
Gold Market Trend Prediction — Streamlit dashboard.

Run via `python main.py` (which launches this with `streamlit run`),
or directly with `streamlit run dashboard.py`.
"""

from __future__ import annotations

import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from config import feature_config as cfg
from pipelines import preprocessing as pp
from pipelines import training as tr
from pipelines import evaluation as ev
from pipelines import prediction as pr
from utils import helpers as hp
from utils import validation as val

st.set_page_config(page_title="Gold Market Prediction System", layout="wide")

# ── session state ────────────────────────────────────────────
if "experiments" not in st.session_state:
    st.session_state.experiments = {}   # feature_group -> {"metrics":..., "package":...}
if "train_raw" not in st.session_state:
    st.session_state.train_raw = None
if "test_raw" not in st.session_state:
    st.session_state.test_raw = None
if "train_labeled" not in st.session_state:
    st.session_state.train_labeled = None
if "test_labeled" not in st.session_state:
    st.session_state.test_labeled = None
if "uploaded_model" not in st.session_state:
    st.session_state.uploaded_model = None
if "uploaded_model_msgs" not in st.session_state:
    st.session_state.uploaded_model_msgs = []


def log_to(container):
    lines = []
    def _log(msg):
        lines.append(str(msg))
        container.code("\n".join(lines))
    return _log


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("CAPSTONE MODEL")

st.sidebar.header("1. Data")
train_file = st.sidebar.file_uploader("Training Dataset (CSV)", type=["csv"], key="train_upload")
test_file = st.sidebar.file_uploader("Testing Dataset (CSV)", type=["csv"], key="test_upload")

st.sidebar.header("2. Saved Model (optional)")
model_file = st.sidebar.file_uploader("Upload Saved Model (.joblib)", type=["joblib", "pkl"], key="model_upload")
if model_file is not None and st.session_state.uploaded_model is None:
    try:
        pkg = hp.load_model_package(io.BytesIO(model_file.getvalue()))
        ok, msgs = val.validate_model_package(pkg)
        st.session_state.uploaded_model = pkg if ok else None
        st.session_state.uploaded_model_msgs = msgs
        if ok:
            st.sidebar.success(f"Model loaded — expects {len(pkg['feature_cols'])} features.")
        else:
            st.sidebar.error("Uploaded model failed validation — see main panel.")
    except Exception as e:
        st.session_state.uploaded_model = None
        st.session_state.uploaded_model_msgs = [f"Could not load file: {e}"]
        st.sidebar.error("Could not load uploaded model file.")

st.sidebar.header("3. Feature Selection")
feature_group = st.sidebar.selectbox(
    "Feature Group",
    list(cfg.FEATURE_GROUPS.keys()),
    index=2,
    help="Technical + Fundamental uses the notebook's actual curated "
         "hybrid feature set (Moozy_Tech) — not a naive union of the "
         "other two groups. See the note on the main page.",
)

st.sidebar.header("4. Model Source")
model_source = st.sidebar.radio(
    "Use which model?",
    ["Train fresh on uploaded data", "Use uploaded saved model"],
    index=0,
)

run_button = st.sidebar.button("Run Model", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MAIN PAGE — HEADER
# ══════════════════════════════════════════════════════════════
st.title("Gold Market Trend Prediction")
st.caption("XAUUSD weekly direction (SHORT / NEUTRAL / LONG) — capstone dashboard")

with st.expander("About the feature groups (read before comparing)", expanded=False):
    st.markdown(
        """
The notebook actually trains on **three distinct** feature sets, not a generic
"technical vs fundamental vs everything" split:

- **Technical** — pure price-action / S/R-zone / trend features (`Tech`, 39 cols).
- **Fundamental** — the full macro release set (`Macro`, 74 cols): NFP, CPI, Core CPI,
  PCE, Core PCE, UNRATE, with release dates, surprises, MoM/YoY.
- **Technical + Fundamental** — the notebook's own hand-curated combined set
  (`Moozy_Tech`, 54 cols). This is **not** a naive union of the two lists above —
  it deliberately drops several macro columns (YoY variants, raw actual/previous
  levels for CPI/PCE) that the author judged redundant, while keeping the full
  technical block plus 3-day return/volatility features not present in the other
  two groups. This is the **only** combined feature set the notebook's XGBoost
  model was ever trained or evaluated on.

No trained model file was supplied with the notebook, so **training fresh** is
the default here — using the notebook's own final training methodology (a
per-class weight-power sweep, chronological validation split, and SHORT/LONG
threshold tuning). You can also upload a `.joblib` produced by a previous run
of this app, or by the notebook's own save cell.
        """
    )

# ══════════════════════════════════════════════════════════════
# DATA VALIDATION
# ══════════════════════════════════════════════════════════════
st.header("1. Data Validation")

col_a, col_b = st.columns(2)
train_ok = test_ok = False

if train_file is not None:
    try:
        train_raw = pp.load_dataset(io.BytesIO(train_file.getvalue()))
        st.session_state.train_raw = train_raw
        ok, msgs = val.validate_raw_dataset(train_raw, "training")
        with col_a:
            st.subheader("Training Dataset")
            st.write(f"{train_raw.shape[0]} rows × {train_raw.shape[1]} columns")
            st.write(f"Date range: {train_raw['date'].min().date()} → {train_raw['date'].max().date()}")
            for m in msgs:
                (st.error if not ok else st.info)(m)
        train_ok = ok
    except Exception as e:
        with col_a:
            st.error(f"Could not read training file: {e}")
else:
    with col_a:
        st.info("Upload a training dataset to begin.")

if test_file is not None:
    try:
        test_raw = pp.load_dataset(io.BytesIO(test_file.getvalue()))
        st.session_state.test_raw = test_raw
        ok, msgs = val.validate_raw_dataset(test_raw, "testing")
        with col_b:
            st.subheader("Testing Dataset")
            st.write(f"{test_raw.shape[0]} rows × {test_raw.shape[1]} columns")
            st.write(f"Date range: {test_raw['date'].min().date()} → {test_raw['date'].max().date()}")
            for m in msgs:
                (st.error if not ok else st.info)(m)
        test_ok = ok
    except Exception as e:
        with col_b:
            st.error(f"Could not read testing file: {e}")
else:
    with col_b:
        st.info("Upload a testing dataset to begin.")

if st.session_state.uploaded_model_msgs:
    st.subheader("Uploaded Model")
    for m in st.session_state.uploaded_model_msgs:
        st.warning(m)

# ══════════════════════════════════════════════════════════════
# FEATURE / MODEL COMPATIBILITY
# ══════════════════════════════════════════════════════════════
st.header("2. Feature Selection & Model Compatibility")

resolved_feats = []
if train_ok:
    ok_fg, msgs_fg, resolved_feats = val.validate_feature_group(
        st.session_state.train_raw, feature_group, "training"
    )
    st.write(f"**Feature Group:** {feature_group}  |  **Resolved model features:** {len(resolved_feats)}")
    for m in msgs_fg:
        (st.error if not ok_fg else st.warning)(m)

    if model_source == "Use uploaded saved model" and st.session_state.uploaded_model is not None:
        model_feats = st.session_state.uploaded_model["feature_cols"]
        compat_ok, compat_msg = val.validate_model_compatibility(resolved_feats, model_feats)
        st.markdown(f"**Model:** uploaded ({type(st.session_state.uploaded_model['model']).__name__})  |  "
                    f"**Expected features:** {len(model_feats)}  |  **Selected features:** {len(resolved_feats)}")
        (st.success if compat_ok else st.error)(compat_msg)
    elif model_source == "Use uploaded saved model":
        st.error("No valid uploaded model available — upload one in the sidebar, or switch to 'Train fresh'.")
    else:
        st.success(f"Will train a fresh XGBoost model on {len(resolved_feats)} features for '{feature_group}'.")
else:
    st.info("Upload and validate a training dataset first.")

# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
st.header("3. Run")

if run_button:
    if not (train_ok and test_ok):
        st.error("Both training and testing datasets must be uploaded and valid before running.")
    else:
        with st.spinner("Labeling datasets (weekly first-touch)..."):
            train_labeled = pp.label_weekly_first_touch_daily(
                st.session_state.train_raw, label_name="Train", verbose=False
            )
            test_labeled = pp.label_weekly_first_touch_daily(
                st.session_state.test_raw, label_name="Test", verbose=False
            )
            st.session_state.train_labeled = train_labeled
            st.session_state.test_labeled = test_labeled

        feats = pp.build_model_feature_cols(feature_group, train_labeled.columns)

        if model_source == "Use uploaded saved model":
            if st.session_state.uploaded_model is None:
                st.error("No valid uploaded model to run.")
                st.stop()
            compat_ok, compat_msg = val.validate_model_compatibility(feats, st.session_state.uploaded_model["feature_cols"])
            if not compat_ok:
                st.error("Cannot run — feature/model mismatch:\n\n" + compat_msg)
                st.stop()
            package = st.session_state.uploaded_model
            # Reorder selection to match the model's training order.
            package = dict(package)
        else:
            log_area = st.empty()
            logger = log_to(log_area)
            with st.spinner(f"Training XGBoost on '{feature_group}' ({len(feats)} features)..."):
                try:
                    package = tr.train_xgboost(train_labeled, feats, log=logger)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
            save_path = hp.model_path_for_group(feature_group)
            saved_at = hp.save_model_package(package, save_path)
            st.success(f"Model trained and saved to `{saved_at}`.")

        with st.spinner("Evaluating on held-out test set..."):
            try:
                metrics = ev.evaluate_model(package, test_labeled)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.session_state.experiments[feature_group] = {
            "metrics": metrics,
            "package": package,
        }
        st.success(f"Run complete for '{feature_group}'.")

# ══════════════════════════════════════════════════════════════
# RESULTS  (most recently run experiment, or selector if several)
# ══════════════════════════════════════════════════════════════
st.header("4. Results")

if not st.session_state.experiments:
    st.info("Run a model to see results here.")
else:
    result_group = st.selectbox(
        "Show results for feature group",
        list(st.session_state.experiments.keys()),
        index=len(st.session_state.experiments) - 1,
    )
    exp = st.session_state.experiments[result_group]
    metrics = exp["metrics"]
    package = exp["package"]

    pred_mode = st.radio(
        "Prediction rule",
        ["Threshold-tuned (recommended)", "Argmax (baseline)"],
        horizontal=True,
        key=f"pred_mode_{result_group}",
    )
    m = metrics["tuned"] if pred_mode.startswith("Threshold") else metrics["argmax"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("ROC-AUC (macro)", f"{m['roc_auc_macro']:.3f}")
    c3.metric("Log Loss", f"{metrics['log_loss']:.3f}")
    c4.metric("Test rows", f"{metrics['n_test_rows']}")

    st.subheader("Confusion Matrix (rows = actual, cols = predicted)")
    cm = m["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=[f"Actual {n}" for n in cfg.CLASS_NAMES],
                          columns=[f"Pred {n}" for n in cfg.CLASS_NAMES])
    st.dataframe(cm_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(cfg.CLASS_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(cfg.CLASS_NAMES, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Classification Report")
    report_df = pd.DataFrame(m["classification_report"]).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

    st.subheader("PR-AUC per class")
    pr_df = pd.DataFrame(m["pr_auc_per_class"]).T
    st.dataframe(pr_df.style.format("{:.3f}"), use_container_width=True)

    # ── Feature importance ──
    if metrics["feature_importance"] is not None:
        st.subheader("Feature Importance (top 10)")
        fi = metrics["feature_importance"].head(10)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(fi.index[::-1], fi.values[::-1], color="#8a6d3b")
        ax2.set_xlabel("Importance")
        fig2.tight_layout()
        st.pyplot(fig2)

    # ── Single prediction ──
    st.subheader("5. Prediction (most recent row in test set)")
    try:
        latest = pr.predict_latest(package, st.session_state.test_labeled)
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Date", str(latest["date"].date()) if latest["date"] is not None else "n/a")
        pc2.metric("Predicted Signal", latest["signal"])
        pc3.metric("Probability", f"{latest['probability']*100:.1f}%")
        st.caption(
            f"Class probabilities — SHORT: {latest['prob_SHORT']*100:.1f}% | "
            f"NEUTRAL: {latest['prob_NEUTRAL']*100:.1f}% | LONG: {latest['prob_LONG']*100:.1f}%"
        )
    except Exception as e:
        st.error(f"Could not produce a prediction: {e}")

# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE — only experiments actually run this session
# ══════════════════════════════════════════════════════════════
st.header("6. Performance Comparison")

if not st.session_state.experiments:
    st.info("Run experiments for one or more feature groups to populate this comparison.")
else:
    rows = []
    for grp, exp in st.session_state.experiments.items():
        mt = exp["metrics"]["tuned"]
        rows.append({
            "Feature Group": grp,
            "Accuracy": mt["accuracy"],
            "Macro Precision": mt["classification_report"]["macro avg"]["precision"],
            "Macro Recall": mt["classification_report"]["macro avg"]["recall"],
            "Macro F1": mt["classification_report"]["macro avg"]["f1-score"],
            "ROC-AUC": mt["roc_auc_macro"],
        })
    comp_df = pd.DataFrame(rows).set_index("Feature Group")
    st.dataframe(comp_df.style.format("{:.3f}"), use_container_width=True)
    st.caption("Only feature groups you have actually run in this session appear here.")
