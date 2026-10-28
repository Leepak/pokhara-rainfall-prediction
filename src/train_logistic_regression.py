
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/pokhara_rainfall_features.csv")
REPORTS_DIR = Path("reports")

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # split_index = int(len(df) * 0.8)
    # train_df = df.iloc[:split_index]
    # test_df = df.iloc[split_index:]

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index - 1].copy()
    test_df = df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Scaling is fitted only on training data.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "model": "Logistic Regression",
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "test_start": str(test_df["date"].min().date()),
        "test_end": str(test_df["date"].max().date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision_rain": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall_rain": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_rain": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        REPORTS_DIR / "logistic_regression_metrics.csv",
        index=False,
    )

    print("=== LOGISTIC REGRESSION ===")
    for key, value in metrics.items():
        print(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No rain", "Rain"],
            zero_division=0,
        )
    )

    print("=== CONFUSION MATRIX ===")
    print("Rows = actual class; columns = predicted class")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nSaved metrics to reports/logistic_regression_metrics.csv")


if __name__ == "__main__":
    main()
