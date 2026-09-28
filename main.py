"""
Master entry point.

    python main.py

What this does, in order:
  1. Validate the project structure (folders/config importable).
  2. If exactly one CSV sits in data/training/ and one in data/testing/,
     run the full pipeline once up front for the default feature group
     (label -> train -> evaluate -> save model) and print a summary —
     this gives you a working model before the dashboard even opens,
     and reproduces the "load training set / load testing set / train
     or load model" flow from the command line.
     If a matching saved model already exists for that feature group,
     it is loaded instead of retrained (pass --force-retrain to
     override).
  3. Launch the Streamlit dashboard (dashboard.py) as a subprocess,
     where you can upload datasets/models and run any feature group
     interactively.

This script does NOT re-run the notebook's own feature engineering
(MT5 pulls, FRED pulls, the S/R zone tracker) — that requires a live
MetaTrader5 terminal and a FRED API key. It expects data/training and
data/testing to contain the notebook's already-engineered daily
dataset (same shape as Dataset_Dashboard.csv / Dataset_Test_Dashboard.csv).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

from config import feature_config as cfg


def validate_project() -> None:
    required_dirs = ["config", "pipelines", "utils", "data/training", "data/testing"]
    missing = [d for d in required_dirs if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"Project structure incomplete — missing directories: {missing}")
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)


def find_single_csv(directory: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if len(matches) == 1:
        return matches[0]
    return None


def run_default_pipeline(feature_group: str, force_retrain: bool) -> None:
    from pipelines import preprocessing as pp
    from pipelines import training as tr
    from pipelines import evaluation as ev
    from utils import helpers as hp

    train_path = find_single_csv("data/training")
    test_path = find_single_csv("data/testing")
    if train_path is None or test_path is None:
        print("No single unambiguous CSV found in data/training/ and data/testing/ "
              "— skipping the CLI pipeline run. Use the dashboard to upload files instead.")
        return

    print(f"Training file : {train_path}")
    print(f"Testing file  : {test_path}")

    model_path = hp.model_path_for_group(feature_group)

    train_raw = pp.load_dataset(train_path)
    test_raw = pp.load_dataset(test_path)

    train_labeled = pp.label_weekly_first_touch_daily(train_raw, label_name="Train")
    test_labeled = pp.label_weekly_first_touch_daily(test_raw, label_name="Test")

    feats = pp.build_model_feature_cols(feature_group, train_labeled.columns)

    if os.path.exists(model_path) and not force_retrain:
        print(f"\nFound existing saved model at {model_path} — loading instead of retraining "
              f"(pass --force-retrain to train a new one).")
        package = hp.load_model_package(model_path)
    else:
        print(f"\nTraining fresh XGBoost model on '{feature_group}' ({len(feats)} features)...")
        package = tr.train_xgboost(train_labeled, feats)
        saved_at = hp.save_model_package(package, model_path)
        print(f"Model saved to: {saved_at}")

    print("\nEvaluating on held-out test set...")
    metrics = ev.evaluate_model(package, test_labeled)
    tuned = metrics["tuned"]
    print(f"\n  Accuracy       : {tuned['accuracy']:.4f}")
    print(f"  ROC-AUC (macro): {tuned['roc_auc_macro']:.4f}")
    print(f"  Log Loss       : {metrics['log_loss']:.4f}")
    print(f"  Test rows      : {metrics['n_test_rows']}")


def launch_dashboard() -> None:
    print("\nLaunching Streamlit dashboard (dashboard.py)...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Market Prediction System — master script")
    parser.add_argument("--feature-group", default="Technical + Fundamental",
                         choices=list(cfg.FEATURE_GROUPS.keys()),
                         help="Feature group for the CLI pre-run (default: Technical + Fundamental).")
    parser.add_argument("--force-retrain", action="store_true",
                         help="Retrain even if a saved model already exists for the chosen feature group.")
    parser.add_argument("--no-dashboard", action="store_true",
                         help="Run the CLI pipeline only; do not launch Streamlit.")
    args = parser.parse_args()

    validate_project()
    run_default_pipeline(args.feature_group, args.force_retrain)

    if not args.no_dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
