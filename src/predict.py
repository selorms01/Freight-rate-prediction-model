"""Generate final predictions using the trained model.

Usage:
    python src/predict.py

Reads:
    models/model.joblib, models/market_reference.joblib
    data/validation.csv, data/validation_predictions_template.csv, data/december_chart_inputs.csv

Writes:
    validation_predictions.csv           (repo root, as required by the assessment)
    outputs/december_chart_inputs_filled.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import prepare_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main():
    model = joblib.load(MODELS / "model.joblib")
    ref = joblib.load(MODELS / "market_reference.joblib")
    daily_ref = ref["daily_ref"]
    fallback_models = ref["fallback_models"]
    fallback_cols = ref["fallback_cols"]
    fallback_min_date = ref["fallback_min_date"]

    # --- validation.csv -> validation_predictions.csv ----------------------
    validation = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")

    X_val = prepare_features(validation, daily_ref, fallback_models, fallback_cols, fallback_min_date)
    validation["predicted_rate"] = model.predict(X_val)

    preds = template[["load_id"]].merge(
        validation[["load_id", "predicted_rate"]], on="load_id", how="left"
    )
    assert preds["predicted_rate"].isna().sum() == 0, "missing predictions for some load_id"
    out_path = ROOT / "validation_predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"Wrote {len(preds):,} predictions to {out_path}")

    # --- december_chart_inputs.csv (fixed lane, no market_index/quote_signal)
    december = pd.read_csv(DATA / "december_chart_inputs.csv")
    X_dec = prepare_features(
        december.drop(columns=["predicted_rate"]), daily_ref, fallback_models, fallback_cols, fallback_min_date
    )
    december["predicted_rate"] = model.predict(X_dec)
    dec_out = OUTPUTS / "december_chart_inputs_filled.csv"
    december.to_csv(dec_out, index=False)
    print(f"Wrote December fixed-lane predictions to {dec_out}")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
