
from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/pokhara_rainfall_features.csv")
OUTPUT_PATH = Path("reports/walk_forward_metrics.csv")

FEATURE_COLUMNS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "pressure_msl_mean",
    "month",
    "day_of_year",
]

# Initial training window: 8 years.
# Each subsequent validation window covers 1 year.
INITIAL_TRAIN_YEARS = 8
VALIDATION_YEARS = 1


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "rain_precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "rain_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "rain_f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df[FEATURE_COLUMNS + ["rain_tomorrow"]].isna().any().any():
        raise ValueError("Missing feature or target values found.")

    first_date = df["date"].min()
    last_date = df["date"].max()

    results = []
    fold = 0

    validation_start = first_date + pd.DateOffset(
        years=INITIAL_TRAIN_YEARS
    )

    while validation_start <= last_date:
        validation_end = validation_start + pd.DateOffset(
            years=VALIDATION_YEARS
        )

        train_df = df[
      df["date"] < validation_start - pd.Timedelta(days=1)
        ].copy()
        validation_df = df[
            (df["date"] >= validation_start)
            & (df["date"] < validation_end)
        ].copy()

        if train_df.empty or validation_df.empty:
            break

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["rain_tomorrow"]
        X_val = validation_df[FEATURE_COLUMNS]
        y_val = validation_df["rain_tomorrow"]

        # Majority-class baseline fitted using training data only.
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_predictions = dummy.predict(X_val)

        # Pipeline fits the scaler only on this fold's training data.
        logistic_model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ])

        logistic_model.fit(X_train, y_train)
        logistic_predictions = logistic_model.predict(X_val)

        fold += 1

        for model_name, predictions in [
            ("Majority baseline", dummy_predictions),
            ("Persistence baseline", (
                validation_df["precipitation_sum"] > 0
            ).astype("int8")),
            ("Logistic regression", logistic_predictions),
        ]:
            row = {
                "fold": fold,
                "model": model_name,
                "train_start": train_df["date"].min().date(),
                "train_end": train_df["date"].max().date(),
                "validation_start": validation_df["date"].min().date(),
                "validation_end": validation_df["date"].max().date(),
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
            }
            row.update(calculate_metrics(y_val, predictions))
            results.append(row)

        print(
            f"Fold {fold}: "
            f"train {train_df['date'].min().date()} to "
            f"{train_df['date'].max().date()} | "
            f"validate {validation_df['date'].min().date()} to "
            f"{validation_df['date'].max().date()}"
        )

        validation_start = validation_end

    if not results:
        raise RuntimeError("No validation folds were generated.")

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "rain_precision",
        "rain_recall",
        "rain_f1",
    ]

    print("\n=== METRICS BY FOLD ===")
    print(results_df[
        ["fold", "model", "validation_start", "validation_end"]
        + metric_columns
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== MEAN METRICS ACROSS FOLDS ===")
    summary = results_df.groupby("model")[metric_columns].mean()
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved detailed results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
