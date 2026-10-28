
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = Path("data/processed/pokhara_rainfall_features.csv")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()

    # Predict tomorrow's rainfall using today's rainfall occurrence.
    predictions = (
        test_df["precipitation_sum"] > 0
    ).astype("int8")

    y_test = test_df["rain_tomorrow"]

    print("=== PERSISTENCE BASELINE ===")
    print(f"Test start: {test_df['date'].min().date()}")
    print(f"Test end:   {test_df['date'].max().date()}")

    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(
        "Rain precision:    "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        "Rain recall:       "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        "Rain F1-score:     "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
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
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
