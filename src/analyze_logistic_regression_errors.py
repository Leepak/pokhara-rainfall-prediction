
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data/processed/pokhara_rainfall_features.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

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

TARGET_COLUMN = "rain_tomorrow"


def main():
    """Analyze logistic regression errors on the chronological test set."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_index = int(len(df) * 0.8)

# Exclude the final training row because its target uses
# rainfall from the first test date.
    train_df = df.iloc[:split_index - 1].copy()
    test_df = df.iloc[split_index:].copy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )

    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[TARGET_COLUMN],
    )

    test_df["actual"] = test_df[TARGET_COLUMN].astype(int)
    test_df["predicted"] = model.predict(test_df[FEATURE_COLUMNS])
    test_df["probability_rain"] = model.predict_proba(
        test_df[FEATURE_COLUMNS]
    )[:, 1]

    # Identify each prediction outcome.
    test_df["error_type"] = "Correct"
    test_df.loc[
        (test_df["actual"] == 1) & (test_df["predicted"] == 1),
        "error_type",
    ] = "True positive"
    test_df.loc[
        (test_df["actual"] == 0) & (test_df["predicted"] == 0),
        "error_type",
    ] = "True negative"
    test_df.loc[
        (test_df["actual"] == 0) & (test_df["predicted"] == 1),
        "error_type",
    ] = "False positive"
    test_df.loc[
        (test_df["actual"] == 1) & (test_df["predicted"] == 0),
        "error_type",
    ] = "False negative"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save all test predictions for reproducible inspection.
    prediction_columns = [
        "date",
        "precipitation_sum",
        "actual",
        "predicted",
        "probability_rain",
        "error_type",
        "month",
    ]
    test_df[prediction_columns].to_csv(
        REPORTS_DIR / "logistic_regression_test_predictions.csv",
        index=False,
    )

    # Save only incorrect predictions.
    errors = test_df[test_df["actual"] != test_df["predicted"]]
    errors[prediction_columns].to_csv(
        REPORTS_DIR / "logistic_regression_errors.csv",
        index=False,
    )

    # Count false positives and false negatives by month.
    monthly_errors = (
        test_df.assign(
            false_positive=(
                (test_df["actual"] == 0)
                & (test_df["predicted"] == 1)
            ).astype(int),
            false_negative=(
                (test_df["actual"] == 1)
                & (test_df["predicted"] == 0)
            ).astype(int),
            actual_rain_days=(test_df["actual"] == 1).astype(int),
        )
        .groupby("month")
        .agg(
            test_days=("actual", "size"),
            actual_rain_days=("actual_rain_days", "sum"),
            false_positives=("false_positive", "sum"),
            false_negatives=("false_negative", "sum"),
        )
        .reset_index()
    )

    monthly_errors["error_count"] = (
        monthly_errors["false_positives"]
        + monthly_errors["false_negatives"]
    )
    monthly_errors["error_rate"] = (
        monthly_errors["error_count"] / monthly_errors["test_days"]
    ).round(4)

    monthly_errors.to_csv(
        REPORTS_DIR / "logistic_regression_monthly_errors.csv",
        index=False,
    )

    tn, fp, fn, tp = confusion_matrix(
        test_df["actual"],
        test_df["predicted"],
        labels=[0, 1],
    ).ravel()

    print("=== ERROR ANALYSIS ===")
    print(f"Training period: {train_df['date'].min().date()} "
          f"to {train_df['date'].max().date()}")
    print(f"Test period: {test_df['date'].min().date()} "
          f"to {test_df['date'].max().date()}")
    print(f"Test days: {len(test_df)}")
    print(f"True positives: {tp}")
    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Total errors: {len(errors)}")

    print("\n=== ERRORS BY MONTH ===")
    print(monthly_errors.to_string(index=False))

    print("\nSaved:")
    print("reports/logistic_regression_test_predictions.csv")
    print("reports/logistic_regression_errors.csv")
    print("reports/logistic_regression_monthly_errors.csv")


if __name__ == "__main__":
    main()
