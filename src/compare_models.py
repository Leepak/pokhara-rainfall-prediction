from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

REPORTS_DIR = Path("reports")

WALK_FORWARD_INPUT = REPORTS_DIR / "walk_forward_metrics.csv"
WALK_FORWARD_OUTPUT = REPORTS_DIR / "model_comparison.csv"

HOLDOUT_INPUT = REPORTS_DIR / "logistic_regression_test_predictions.csv"
HOLDOUT_OUTPUT = REPORTS_DIR / "holdout_model_comparison_exploratory.csv"

ACTIVE_MODELS = {
    "Majority baseline",
    "Persistence baseline",
    "Logistic regression",
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "rain_precision",
    "rain_recall",
    "rain_f1",
]


def calculate_metrics(actual, predicted):
    """Calculate binary classification metrics for rain as the positive class."""
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "rain_precision": precision_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "rain_recall": recall_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "rain_f1": f1_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
    }


def create_walk_forward_comparison():
    """Summarize mean metrics across annual validation folds."""
    if not WALK_FORWARD_INPUT.exists():
        raise FileNotFoundError(
            f"Missing {WALK_FORWARD_INPUT}. Run walk_forward_validation.py first."
        )

    df = pd.read_csv(WALK_FORWARD_INPUT)

    required = {"model", "fold", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in walk-forward data: {sorted(missing)}")

    # Keep only active models; archived LSTM results are excluded.
    df = df[df["model"].isin(ACTIVE_MODELS)].copy()

    if df.empty:
        raise ValueError("No active model results found in walk-forward metrics.")

    comparison = (
        df.groupby("model", as_index=False)[METRICS]
        .mean()
        .sort_values("model")
    )

    comparison.to_csv(WALK_FORWARD_OUTPUT, index=False)

    print("\n=== PRIMARY COMPARISON: WALK-FORWARD VALIDATION ===")
    print("Values are means across annual validation folds.")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved to: {WALK_FORWARD_OUTPUT}")


def create_exploratory_holdout_comparison():
    """Recalculate holdout metrics consistently from saved test predictions."""
    if not HOLDOUT_INPUT.exists():
        raise FileNotFoundError(
            f"Missing {HOLDOUT_INPUT}. Run train_logistic_regression.py first."
        )

    df = pd.read_csv(HOLDOUT_INPUT)

    required = {"date", "precipitation_sum", "actual", "predicted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in holdout predictions: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    actual = df["actual"].astype(int)

    # Majority baseline: always predict rain, matching the saved baseline.
    always_rain = pd.Series(1, index=df.index)

    # Persistence: today's rain status is used to predict tomorrow's status.
    persistence = (df["precipitation_sum"] > 0).astype(int)

    predictions = {
        "Always-rain baseline": always_rain,
        "Persistence baseline": persistence,
        "Logistic regression": df["predicted"].astype(int),
    }

    rows = []
    for model_name, predicted in predictions.items():
        rows.append({
            "model": model_name,
            **calculate_metrics(actual, predicted),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(HOLDOUT_OUTPUT, index=False)

    print("\n=== EXPLORATORY COMPARISON: PREVIOUSLY INSPECTED HOLDOUT ===")
    print(f"Test period: {df['date'].min().date()} to {df['date'].max().date()}")
    print("These results are exploratory, not an untouched final evaluation.")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved to: {HOLDOUT_OUTPUT}")


def main():
    create_walk_forward_comparison()
    create_exploratory_holdout_comparison()


if __name__ == "__main__":
    main()
