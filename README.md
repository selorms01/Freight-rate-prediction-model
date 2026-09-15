# Freight Rate Prediction

Predicts `posted_rate` for freight loads from lane, load, and market features.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
# 1. Clean data, engineer features, run a time-based holdout evaluation,
#    and fit the final model on all labeled data.
python src/train.py

# 2. Generate validation_predictions.csv (12,000 loads) and fill in
#    outputs/december_chart_inputs_filled.csv (fixed lane, 31 days).
python src/predict.py

# 3. Validate the two output files and render the December chart
#    (provided by the assessment).
python score.py --predictions validation_predictions.csv \
                 --december-predictions outputs/december_chart_inputs_filled.csv
```

`train.py` writes `models/model.joblib`, `models/market_reference.joblib`, and
holdout diagnostics to `reports/`. `predict.py` writes `validation_predictions.csv`
to the repo root (as required) and the completed December file to `outputs/`.

## Project layout

```
data/           input CSVs (as provided)
src/
  features.py   shared cleaning + feature engineering (used by train & predict)
  train.py      time-based holdout evaluation + final model fit
  predict.py    generates the two required prediction files
models/         saved model + market-signal reference table
reports/        holdout metrics (json) and diagnostic plots
outputs/        completed December fixed-lane file
score.py        provided scorer (validates outputs, draws the December chart)
```

## Approach summary

See `reports/model_report.docx` for the full writeup (data quality issues,
validation strategy, model choice, and results). In short: gradient-boosted
trees (`HistGradientBoostingRegressor`, MAE loss) on distance, equipment,
lane (pickup/delivery), weight, market_index, quote_signal, and date-derived
features, evaluated on a time-based holdout (Sep-Oct 2025) to mimic
forecasting forward into Nov-Dec, then refit on all labeled data for the
final predictions.
