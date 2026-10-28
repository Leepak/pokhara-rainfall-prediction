
from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/pokhara_daily_weather.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "pokhara_rainfall_features.csv"

RAIN_THRESHOLD_MM = 0.0


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}. "
            "Run src/download_data.py first."
        )

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Check data integrity.
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in the raw dataset.")

    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )

    if not df["date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError(
            "Dates are missing or irregular. Resolve this before "
            "creating next-day targets."
        )

    # Target: 1 if it rains tomorrow, otherwise 0.
    df["rain_tomorrow"] = (
        df["precipitation_sum"].shift(-1) > RAIN_THRESHOLD_MM
    ).astype("int8")

    # Calendar features available on the prediction date.
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year

    # Remove the final row because tomorrow's rainfall is unknown.
    df = df.iloc[:-1].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset preparation completed.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")
    print(
        df["rain_tomorrow"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No rain", 1: "Rain"})
        .to_string()
    )

    print("\nTarget proportions:")
    print(
        df["rain_tomorrow"]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
    )

    print("\nPreview:")
    print(df[["date", "precipitation_sum", "rain_tomorrow"]].head())


if __name__ == "__main__":
    main()
