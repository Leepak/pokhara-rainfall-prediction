
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/pokhara_daily_weather.csv")
FIGURES_DIR = Path("figures")

sns.set_theme(style="whitegrid")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run src/download_data.py first."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    rainfall = "precipitation_sum"

    # 1. Basic data quality
    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['date'].min().date()} "
          f"to {df['date'].max().date()}")
    print(f"Duplicate dates: {df['date'].duplicated().sum()}")
    print(f"Missing values:\n{df.isna().sum()}")

    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )
    missing_dates = expected_dates.difference(df["date"])

    print(f"Missing calendar dates: {len(missing_dates)}")

    # 2. Descriptive statistics
    print("\n=== WEATHER SUMMARY ===")
    print(df.drop(columns="date").describe().round(2).T)

    # 3. Rainfall distribution
    print("\n=== RAINFALL ANALYSIS ===")
    print(f"Mean daily rainfall: {df[rainfall].mean():.2f} mm")
    print(f"Median daily rainfall: {df[rainfall].median():.2f} mm")
    print(f"Maximum daily rainfall: {df[rainfall].max():.2f} mm")
    print(f"Days with rainfall > 0 mm: "
          f"{(df[rainfall] > 0).sum()}")
    print(f"Days with rainfall >= 1 mm: "
          f"{(df[rainfall] >= 1).sum()}")
    print(f"Days with rainfall >= 10 mm: "
          f"{(df[rainfall] >= 10).sum()}")

    # 4. Daily rainfall over time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["date"], df[rainfall], linewidth=0.6)
    ax.set(
        title="Daily Precipitation in Pokhara (2010–2025)",
        xlabel="Date",
        ylabel="Precipitation (mm)",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "daily_rainfall.png",
        dpi=150,
    )
    plt.close(fig)

    # 5. Monthly average rainfall
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")[rainfall].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(kind="bar", ax=ax)
    ax.set(
        title="Average Daily Rainfall by Month",
        xlabel="Month",
        ylabel="Average daily precipitation (mm)",
    )
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        rotation=0,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "monthly_rainfall.png",
        dpi=150,
    )
    plt.close(fig)

    # 6. Annual precipitation totals
    annual = df.groupby(df["date"].dt.year)[rainfall].sum()

    print("\n=== ANNUAL PRECIPITATION TOTALS (mm) ===")
    print(annual.round(1).to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    annual.plot(kind="bar", ax=ax)
    ax.set(
        title="Annual Precipitation in Pokhara",
        xlabel="Year",
        ylabel="Total precipitation (mm)",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "annual_rainfall.png",
        dpi=150,
    )
    plt.close(fig)

    # 7. Export descriptive summaries
    summary = df.drop(columns=["date", "month"]).describe().T
    summary.to_csv("reports/descriptive_statistics.csv")

    print("\nFigures saved in:", FIGURES_DIR)
    print("Statistics saved in reports/descriptive_statistics.csv")
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
