"""Train and validate the freight rate model.

Usage:
    python src/train.py

Writes:
    models/model.joblib                 -- final model, refit on all labeled data
    models/market_reference.joblib      -- daily market-signal lookup + fallback regressors
    reports/validation_metrics.json     -- holdout metrics
    reports/pred_vs_actual.png          -- diagnostic plot
    reports/feature_importance.png      -- permutation importance plot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    build_daily_market_reference,
    estimate_missing_market_signal,
    prepare_features,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def main():
    train_test = pd.read_csv(DATA / "train_test.csv")
    validation_raw = pd.read_csv(DATA / "validation.csv")

    # --- daily market-signal reference, used to impute missing market_index /
    # quote_signal and (later) to estimate them for the December fixed-lane file.
    # Built from BOTH files: both are fully observed historical/labeled-feature
    # data available before we generate any predictions, so this isn't leakage
    # of the *target* -- it only borrows same-day values across different loads.
    daily_ref = build_daily_market_reference(train_test, validation_raw)
    fallback_models, fallback_cols, fallback_min_date = estimate_missing_market_signal(daily_ref)
    joblib.dump(
        {
            "daily_ref": daily_ref,
            "fallback_models": fallback_models,
            "fallback_cols": fallback_cols,
            "fallback_min_date": fallback_min_date,
        },
        MODELS / "market_reference.joblib",
    )

    # For the *internal* time-based validation below we deliberately rebuild a
    # reference using ONLY train_test.csv, so the holdout evaluation reflects
    # what would genuinely be known before validation.csv existed.
    daily_ref_tt = build_daily_market_reference(train_test)
    fb_models_tt, fb_cols_tt, fb_min_tt = estimate_missing_market_signal(daily_ref_tt)

    X_all = prepare_features(train_test, daily_ref_tt, fb_models_tt, fb_cols_tt, fb_min_tt)
    y_all = train_test["posted_rate"].astype(float)
    dates = pd.to_datetime(train_test["date"])

    # --- time-based split -------------------------------------------------
    # train_test.csv spans 2025-01-01..2025-10-31 and we must forecast
    # 2025-11-01..2025-12-31. A random split would let the model "peek" at
    # market conditions from the same week it's tested on. Instead we hold
    # out the most recent ~2 months (Sep+Oct) to mimic forecasting forward
    # in time, which is the actual task.
    cutoff = pd.Timestamp("2025-09-01")
    train_mask = dates < cutoff
    holdout_mask = ~train_mask

    cat_idx = [ALL_FEATURE_COLS.index(c) for c in CATEGORICAL_COLS]
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=8,
        learning_rate=0.06,
        max_iter=500,
        l2_regularization=0.1,
        categorical_features=cat_idx,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_all[train_mask], y_all[train_mask])

    pred_holdout = model.predict(X_all[holdout_mask])
    y_holdout = y_all[holdout_mask]

    metrics = {
        "n_train": int(train_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "holdout_period": "2025-09-01 to 2025-10-31 (time-based holdout within train_test.csv)",
        "MAE": mean_absolute_error(y_holdout, pred_holdout),
        "RMSE": rmse(y_holdout, pred_holdout),
        "MAPE": float(mean_absolute_percentage_error(y_holdout, pred_holdout)),
        "R2": float(r2_score(y_holdout, pred_holdout)),
    }
    print("Time-based holdout metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 5-fold-ish check: also report metrics on a random 15% split, for
    # comparison -- expected to look optimistic vs. the time-based split.
    rng = np.random.RandomState(0)
    rand_holdout = rng.rand(len(X_all)) < 0.15
    model_rand = HistGradientBoostingRegressor(
        loss="absolute_error", max_depth=8, learning_rate=0.06, max_iter=500,
        l2_regularization=0.1, categorical_features=cat_idx,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    ).fit(X_all[~rand_holdout], y_all[~rand_holdout])
    pred_rand = model_rand.predict(X_all[rand_holdout])
    metrics["random_split_MAE"] = mean_absolute_error(y_all[rand_holdout], pred_rand)
    metrics["random_split_R2"] = float(r2_score(y_all[rand_holdout], pred_rand))
    print(f"  (for comparison) random-split MAE: {metrics['random_split_MAE']:.2f}, "
          f"R2: {metrics['random_split_R2']:.4f}")

    with open(REPORTS / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- diagnostics --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_holdout, pred_holdout, s=4, alpha=0.25, color="#064A56")
    lims = [0, max(y_holdout.max(), pred_holdout.max())]
    ax.plot(lims, lims, color="red", linewidth=1)
    ax.set_xlabel("Actual posted_rate ($)")
    ax.set_ylabel("Predicted posted_rate ($)")
    ax.set_title("Holdout (Sep-Oct 2025): Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(REPORTS / "pred_vs_actual.png", dpi=150)
    plt.close(fig)

    perm = permutation_importance(
        model, X_all[holdout_mask], y_holdout, n_repeats=5, random_state=42, n_jobs=-1
    )
    order = np.argsort(perm.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(ALL_FEATURE_COLS)[order], perm.importances_mean[order], color="#064A56")
    ax.set_xlabel("Permutation importance (drop in R2)")
    ax.set_title("Feature importance (holdout set)")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    # --- refit on ALL labeled data for the model we actually ship ---------
    final_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=8,
        learning_rate=0.06,
        max_iter=model.n_iter_,  # use the iteration count chosen by early stopping above
        l2_regularization=0.1,
        categorical_features=cat_idx,
        random_state=42,
    )
    final_model.fit(X_all, y_all)
    joblib.dump(final_model, MODELS / "model.joblib")
    print(f"\nFinal model refit on all {len(X_all):,} labeled rows "
          f"({model.n_iter_} boosting iterations) and saved to models/model.joblib")


if __name__ == "__main__":
    main()
