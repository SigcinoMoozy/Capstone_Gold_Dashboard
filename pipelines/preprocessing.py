"""
Preprocessing pipeline.

Ported directly from the Capstone notebook. The notebook's own feature
engineering (MT5 pulls, FRED pulls, the event-driven S/R zone tracker,
`add_all_features`) requires a live MetaTrader5 terminal + a FRED API
key, so it is NOT re-run here — this app picks up the pipeline exactly
where the notebook's own dashboard-ready CSVs picked up: an already
feature-engineered daily dataset with no target column yet.

What this module DOES reproduce exactly from the notebook:
  - label_weekly_first_touch_daily(): the weekly first-touch labeling
    logic (Monday-open reference, scan day by day, censor rows after
    the trigger day to avoid leakage).
  - Feature-group column selection (Technical / Fundamental /
    Technical + Fundamental) against config/feature_config.py.
  - dtype downcasting used before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import feature_config as cfg


# ══════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════

def load_dataset(path_or_buffer) -> pd.DataFrame:
    """Load a CSV (path or file-like/uploaded buffer) and parse dates."""
    df = pd.read_csv(path_or_buffer)
    if "date" not in df.columns:
        raise ValueError(
            "Uploaded file has no 'date' column. This pipeline expects "
            "the notebook's engineered daily dataset (date, OHLCV, "
            "technical + fundamental feature columns)."
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# LABELING — weekly first-touch, daily rows preserved (censored)
# Ported verbatim (logic-for-logic) from the notebook.
# ══════════════════════════════════════════════════════════════

def label_weekly_first_touch_daily(
    df: pd.DataFrame,
    date_col: str = "date",
    up: float = cfg.LABEL_UP_THRESHOLD,
    down: float = cfg.LABEL_DOWN_THRESHOLD,
    min_days_in_week: int = cfg.LABEL_MIN_DAYS_IN_WEEK,
    label_name: str = "Dataset",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    One label per ISO week (target in {-1, 0, 1} = SHORT/NEUTRAL/LONG),
    broadcast onto every daily row UP TO AND INCLUDING the day the
    threshold first touched (rows after the trigger day are dropped —
    keeping them would leak the outcome that created the label into
    the feature set). Weeks that never trigger keep every day (NO
    TRADE stays genuinely undecided the whole week).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    iso = out[date_col].dt.isocalendar()
    out["_iso_year"] = iso["year"]
    out["_iso_week"] = iso["week"]

    results = []
    kept_rows = []
    same_day_double_touch = 0

    for (yr, wk), grp in out.groupby(["_iso_year", "_iso_week"], sort=False):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        n_days = len(grp)
        week_open = grp.iloc[0]["open"]
        entry_date = grp.iloc[0][date_col]
        exit_date = grp.iloc[-1][date_col]

        label = 0
        trigger_date = pd.NaT
        trigger_day_offset = None

        for offset, row in grp.iterrows():
            up_hit = (row["high"] - week_open) / week_open >= up
            down_hit = (row["low"] - week_open) / week_open <= down

            if up_hit and down_hit:
                same_day_double_touch += 1
                day_return = row["close"] - row["open"]
                label = 1 if day_return >= 0 else -1
                trigger_date, trigger_day_offset = row[date_col], offset
                break
            elif up_hit:
                label = 1
                trigger_date, trigger_day_offset = row[date_col], offset
                break
            elif down_hit:
                label = -1
                trigger_date, trigger_day_offset = row[date_col], offset
                break

        results.append({
            "_iso_year": yr, "_iso_week": wk,
            "week_open": week_open, "n_days": n_days,
            "entry_date": entry_date, "exit_date": exit_date,
            "target": label, "trigger_date": trigger_date,
            "trigger_day_offset": trigger_day_offset,
        })

        if n_days < min_days_in_week:
            continue

        cutoff = trigger_day_offset if trigger_day_offset is not None else (n_days - 1)
        week_rows = grp.iloc[: cutoff + 1].copy()
        week_rows["_iso_year"], week_rows["_iso_week"] = yr, wk
        kept_rows.append(week_rows)

    week_summary = pd.DataFrame(results)
    short_weeks = (week_summary["n_days"] < min_days_in_week).sum()
    week_summary_valid = week_summary[week_summary["n_days"] >= min_days_in_week].copy()

    dropped_final_week = False
    if len(week_summary_valid) > 0:
        last_date_in_df = out[date_col].max()
        if week_summary_valid.iloc[-1]["exit_date"] == last_date_in_df:
            typical_days = week_summary_valid["n_days"].mode().iloc[0]
            if week_summary_valid.iloc[-1]["n_days"] < typical_days:
                last_yr, last_wk = week_summary_valid.iloc[-1][["_iso_year", "_iso_week"]]
                week_summary_valid = week_summary_valid.iloc[:-1]
                kept_rows = [
                    r for r in kept_rows
                    if not ((r["_iso_year"].iloc[0] == last_yr) and (r["_iso_week"].iloc[0] == last_wk))
                ]
                dropped_final_week = True

    daily_out = pd.concat(kept_rows, ignore_index=True) if kept_rows else out.iloc[0:0].copy()

    merge_cols = ["_iso_year", "_iso_week", "week_open", "n_days",
                  "entry_date", "exit_date", "target", "trigger_date", "trigger_day_offset"]
    daily_out = daily_out.merge(week_summary_valid[merge_cols], on=["_iso_year", "_iso_week"], how="inner")
    daily_out = daily_out.drop(columns=["_iso_year", "_iso_week"]).reset_index(drop=True)

    if verbose:
        print(f"  [{label_name}] weeks found: {len(week_summary)}  "
              f"| kept: {len(week_summary_valid)}  | dropped short weeks: {short_weeks}"
              f"{'  | dropped in-progress final week' if dropped_final_week else ''}")
        print(f"  [{label_name}] daily rows kept (censored at trigger day): {len(daily_out)}")
        counts = daily_out["target"].value_counts().sort_index()
        total = len(daily_out)
        for cls, cnt in counts.items():
            lbl = {-1: "SELL/SHORT (-1)", 0: "NO TRADE (0)", 1: "BUY/LONG (1)"}[cls]
            pct = cnt / total * 100 if total else 0
            print(f"    {lbl:<16}: {cnt:>6}  ({pct:.1f}%)")

    return daily_out


# ══════════════════════════════════════════════════════════════
# FEATURE GROUP SELECTION
# ══════════════════════════════════════════════════════════════

def get_feature_group_columns(group_name: str, available_columns) -> list[str]:
    """
    Return the feature-group's defined columns that also exist in the
    dataframe (intersection, preserving the config's declared order).
    """
    if group_name not in cfg.FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group '{group_name}'. "
                          f"Choose from: {list(cfg.FEATURE_GROUPS)}")
    wanted = cfg.FEATURE_GROUPS[group_name]
    available = set(available_columns)
    return [c for c in wanted if c in available]


def build_model_feature_cols(group_name: str, df_columns) -> list[str]:
    """
    The actual X columns used for training/prediction: the feature
    group's columns MINUS whatever must always be excluded (raw
    OHLCV, date, target, leakage-prone weekly-labeling columns, raw
    release-date timestamps).
    """
    group_cols = get_feature_group_columns(group_name, df_columns)
    drop_set = set(cfg.BASE_DROP_COLS)
    return [c for c in group_cols if c not in drop_set]


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]):
    """
    Drop rows with missing/non-finite target, split into X (downcast,
    feature_cols only) and y (int target in {-1,0,1}).
    """
    d = df.copy()
    before = len(d)
    d = d[d["target"].notna()]
    d = d[np.isfinite(d["target"].astype(float))]
    d = d.reset_index(drop=True)

    missing = [c for c in feature_cols if c not in d.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

    X = downcast(d[feature_cols].copy())
    y = d["target"].astype(int)
    return X, y, before, len(d)
