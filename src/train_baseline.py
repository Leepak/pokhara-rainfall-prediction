
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = Path("data/processed/pokhara_rainfall_features.csv")
REPORTS_DIR = Path("reports")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run src/prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Reserve the latest 20% of dates for testing.
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index - 1].copy()
    test_df = df.iloc[split_index:].copy()

    y_train = train_df["rain_tomorrow"]
    y_test = test_df["rain_tomorrow"]

    # A baseline that always predicts the majority training class.
    model = DummyClassifier(strategy="most_frequent")
    model.fit([[0]] * len(y_train), y_train)

    predictions = model.predict([[0]] * len(y_test))

    metrics = {
        "model": "Always predict majority class",
        "train_start": str(train_df["date"].min().date()),
        "train_end": str(train_df["date"].max().date()),
        "test_start": str(test_df["date"].min().date()),
        "test_end": str(test_df["date"].max().date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_rain_days": int(y_test.sum()),
        "test_no_rain_days": int((y_test == 0).sum()),
        "accuracy": accuracy_score(y_test, predictions),
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
        REPORTS_DIR / "baseline_metrics.csv", index=False
    )

    print("=== BASELINE MODEL ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

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

    print("\nSaved metrics to reports/baseline_metrics.csv")


if __name__ == "__main__":
    main()
