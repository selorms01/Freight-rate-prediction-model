"""Shared data-cleaning and feature-engineering logic for the freight rate model.

Kept in one module so training and prediction always apply identical
transformations (avoids train/serve skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["pickup", "delivery", "equipment"]
NUMERIC_COLS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
]
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def clean_weight(series: pd.Series) -> pd.Series:
    """Weight has a small number of negative values (~0.6% of rows) that are
    almost certainly a sign/entry error rather than a different physical
    quantity -- the magnitudes fall in the same 0-47,500 lb range as the
    positive values. We take the absolute value rather than dropping rows.
    """
    return series.abs()


def build_daily_market_reference(*frames: pd.DataFrame) -> pd.DataFrame:
    """market_index and quote_signal behave like a shared, date-level market
    signal (weekly seasonality + a slower drift) with small per-load noise
    layered on top -- the daily mean across hundreds of loads is a stable
    estimate of that day's signal. We build one reference table (date ->
    mean market_index, mean quote_signal) from every frame that actually
    carries observed values for that date (train_test + validation), which
    lets us:
      1. impute the small number of missing values within those frames, and
      2. estimate the signal for december_chart_inputs.csv, which does not
         carry market_index/quote_signal columns at all but does share
         dates with validation.csv.
    """
    parts = []
    for frame in frames:
        if "market_index" not in frame.columns:
            continue
        parts.append(frame[["date", "market_index", "quote_signal"]])
    stacked = pd.concat(parts, ignore_index=True)
    daily = stacked.groupby("date")[["market_index", "quote_signal"]].mean()
    return daily


def estimate_missing_market_signal(daily_ref: pd.DataFrame) -> pd.DataFrame:
    """Fit a light day-of-week + smooth annual-seasonality regression on the
    daily reference table, used only to fill in dates that have zero observed
    rows anywhere (shouldn't happen for Nov/Dec 2025 here, but keeps the
    pipeline robust rather than failing on an edge case).
    """
    ref = daily_ref.reset_index().copy()
    ref["date"] = pd.to_datetime(ref["date"])
    ref["doy"] = ref["date"].dt.dayofyear
    ref["dow"] = ref["date"].dt.dayofweek
    X = pd.get_dummies(ref["dow"], prefix="dow", drop_first=True)
    X["sin1"] = np.sin(2 * np.pi * ref["doy"] / 365.25)
    X["cos1"] = np.cos(2 * np.pi * ref["doy"] / 365.25)
    X["t"] = (ref["date"] - ref["date"].min()).dt.days
    from sklearn.linear_model import LinearRegression

    models = {}
    for col in ["market_index", "quote_signal"]:
        model = LinearRegression().fit(X, ref[col])
        models[col] = model
    return models, X.columns.tolist(), ref["date"].min()


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def prepare_features(
    df: pd.DataFrame,
    daily_ref: pd.DataFrame,
    fallback_models=None,
    fallback_cols=None,
    fallback_min_date=None,
) -> pd.DataFrame:
    """Apply cleaning + feature engineering to a raw dataframe, returning a
    frame with exactly ALL_FEATURE_COLS ready to feed to the model.
    df must contain: pickup, delivery, distance, equipment, weight, date.
    market_index / quote_signal are optional -- if absent (december chart
    inputs) or missing per-row, they're filled from daily_ref by date, with
    a regression fallback for any date not present in daily_ref at all.
    """
    df = add_date_features(df)
    df["weight"] = clean_weight(df["weight"])
    df["weight"] = df["weight"].fillna(df.groupby("equipment")["weight"].transform("median"))

    ref = daily_ref.copy()
    ref.index = pd.to_datetime(ref.index)

    for col in ["market_index", "quote_signal"]:
        if col not in df.columns:
            df[col] = np.nan
        merged = df["date"].map(ref[col])
        df[col] = df[col].fillna(merged)

    still_missing = df["market_index"].isna() | df["quote_signal"].isna()
    if still_missing.any() and fallback_models is not None:
        sub = df.loc[still_missing, "date"]
        doy = sub.dt.dayofyear
        dow = sub.dt.dayofweek
        Xf = pd.get_dummies(dow, prefix="dow", drop_first=True)
        for c in fallback_cols:
            if c not in Xf.columns and c.startswith("dow_"):
                Xf[c] = 0
        Xf["sin1"] = np.sin(2 * np.pi * doy / 365.25)
        Xf["cos1"] = np.cos(2 * np.pi * doy / 365.25)
        Xf["t"] = (sub - fallback_min_date).dt.days
        Xf = Xf.reindex(columns=fallback_cols, fill_value=0)
        for col in ["market_index", "quote_signal"]:
            pred = fallback_models[col].predict(Xf)
            df.loc[still_missing, col] = df.loc[still_missing, col].fillna(
                pd.Series(pred, index=sub.index)
            )

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df[ALL_FEATURE_COLS]
