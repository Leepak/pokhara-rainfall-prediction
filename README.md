# Pokhara Rainfall Prediction

## Project overview

This project investigates whether historical daily weather variables can
support next-day rainfall-occurrence prediction for Pokhara, Nepal.

It is an exploratory, reproducible machine-learning study and a foundation
for broader research into environmental prediction and disaster-risk
decision support. It is not an operational flood forecasting or warning
system.

## Research question

How do simple baseline methods and logistic regression compare when
predicting whether rainfall will occur on the following day using historical
daily weather data?

## Study area and data

- Location: Pokhara, Nepal (approximately 28.21 N, 83.99 E).
- Period: 1 January 2010 to 31 December 2025.
- Source: Open-Meteo Historical Weather API.
- Temporal resolution: daily.
- Raw observations: 5,844 rows.
- Prediction records: 5,843 rows after constructing the next-day target.

The raw dataset includes daily precipitation, mean/minimum/maximum
temperature, mean relative humidity, mean wind speed, and mean sea-level
pressure.

The dataset represents gridded historical weather data and should not be
assumed to be equivalent to measurements from a local rain gauge. Verify
the precise data product and retrieval settings in the data-download script
before making scientific claims about observation quality.

## Target and features

The target, `rain_tomorrow`, is defined as:

- `1`: the following day's precipitation is greater than 0 mm.
- `0`: the following day's precipitation is 0 mm.

The final date is removed because its next-day target is unavailable.
The current-day weather variables, month, and day of year are used as
features by the logistic-regression workflow.

This is a binary rainfall-occurrence task. It does not predict rainfall
amount, river discharge, flood occurrence, flood severity, or downstream
exposure.

## Models evaluated

1. Majority-class baseline.
2. Persistence baseline, which uses today's rainfall occurrence to predict
   tomorrow's rainfall occurrence.
3. Logistic regression with standardized features.

The active comparison uses chronological evaluation and expanding-window
walk-forward validation. An earlier LSTM experiment is archived and is not
part of the active model comparison.

## Main findings

Across eight annual walk-forward validation folds, the recorded mean
metrics were:

| Model | Accuracy | Balanced accuracy | Rain precision | Rain recall | Rain F1 |
|---|---:|---:|---:|---:|---:|
| Persistence baseline | 0.8302 | 0.8273 | 0.8375 | 0.8375 | 0.8375 |
| Logistic regression | 0.8223 | 0.8137 | 0.8082 | 0.8744 | 0.8386 |
| Majority baseline | 0.5258 | 0.5000 | 0.5258 | 1.0000 | 0.6873 |

The persistence baseline has the highest mean accuracy and balanced accuracy.
Logistic regression has slightly higher mean rainfall F1 and recall.
These results illustrate the importance of comparing multiple metrics and
simple baselines rather than assuming a more complex model will perform
better.

The original chronological holdout period (19 October 2022 to
30 December 2025) has already been inspected and used for exploratory
analysis, including threshold exploration. Its results must not be described
as an untouched final test evaluation.

## Project structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── reports/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Use a supported Python environment. The project scripts have been run in
Python 3.14.7 in the development environment.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproduce the current workflow

Run these commands from the project root, with the virtual environment
activated:

```bash
python src/explore_data.py
python src/prepare_data.py
python src/train_baseline.py
python src/train_persistence_baseline.py
python src/train_logistic_regression.py
python src/walk_forward_validation.py
python src/compare_walk_forward_models.py
python src/analyze_logistic_regression_errors.py
```

These scripts generate descriptive statistics, figures, model metrics,
walk-forward comparisons, and error-analysis reports.

The workflow above uses the existing raw CSV. To retrieve data again, inspect
`src/download_data.py` and its API settings first; new retrievals may differ
from the dataset used in the recorded experiments.

## Limitations

- Only one location is modeled; this is not a spatial or spatiotemporal
  multi-location study.
- The target is next-day rainfall occurrence, not flash-flood occurrence.
- Historical gridded weather data may differ from local sensor or rain-gauge
  observations.
- The original holdout period has been repeatedly inspected, so its results
  are exploratory rather than a final untouched estimate.
- Walk-forward results vary by year and do not establish future operational
  performance.
- No live forecast feed, IoT stream, river model, downstream exposure
  assessment, or alert-delivery workflow is implemented.
- The current project is a research prototype, not a safety-critical
  warning service.

## Future direction

Potential extensions include evaluating forecast products, incorporating
reliable local observations and flood-event records, extending the analysis
to multiple locations, assessing downstream exposure with geospatial data,
and building a dashboard with simulated location-specific alerts. Each
extension depends on data availability and appropriate validation.

## Research integrity

This project prioritizes transparent methods, reproducible experiments,
comparison with simple baselines, and explicit limitations. Results should
be reported as generated by the documented workflow, and future improvements
should be evaluated on data not used to select the model or its threshold.
