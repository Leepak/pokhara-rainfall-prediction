
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

FILES = [
    REPORTS_DIR / "walk_forward_metrics.csv",
]

OUTPUT_FILE = REPORTS_DIR / "walk_forward_model_comparison.csv"


def main():
    """Combine walk-forward validation results for the active models."""
    dataframes = []

    for file_path in FILES:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required results file not found: {file_path}"
            )

        df = pd.read_csv(file_path)
        dataframes.append(df)

    comparison = pd.concat(dataframes, ignore_index=True)

    # Summarize performance across validation years for each model.
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "rain_precision",
        "rain_recall",
        "rain_f1",
    ]

    required_columns = {"model", *metric_columns}
    missing_columns = required_columns - set(comparison.columns)

    if missing_columns:
        raise ValueError(
            f"Missing expected columns in validation results: "
            f"{sorted(missing_columns)}"
        )

    summary = (
        comparison.groupby("model")[metric_columns]
        .mean()
        .reset_index()
        .sort_values("balanced_accuracy", ascending=False)
        .round(4)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_FILE, index=False)

    print("\nWalk-forward model comparison")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
