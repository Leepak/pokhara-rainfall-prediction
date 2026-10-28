
from pathlib import Path

import pandas as pd
import requests

LATITUDE = 28.21
LONGITUDE = 83.99

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "precipitation_sum",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "pressure_msl_mean",
    ],
    "timezone": "Asia/Kathmandu",
    "precipitation_unit": "mm",
    "wind_speed_unit": "kmh",
}

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "pokhara_daily_weather.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Requesting historical weather data...")

    response = requests.get(
        API_URL,
        params=PARAMS,
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()

    if "daily" not in payload:
        raise ValueError(f"Unexpected API response: {payload}")

    df = pd.DataFrame(payload["daily"])
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])

    if df.empty:
        raise ValueError("The API returned an empty dataset.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in the response.")

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload successful.")
    print(f"Rows: {len(df)}")
    print(f"First date: {df['date'].min().date()}")
    print(f"Last date: {df['date'].max().date()}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nDataset preview:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
