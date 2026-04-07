import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Path to the original data
ORIGINAL_CSV = os.path.join(os.path.dirname(__file__), "../premier_league.csv")
# Path to the output synthetic data
OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__), "../additional_data/additional_premier_league.csv"
)


def generate_synthetic_rows(df, n_rows):
    # For each column, get value ranges or unique values
    synthetic_rows = []
    teams = df["Team"].unique()
    opponents = df["Opponent"].unique()
    referees = df["Referee"].unique()
    formations = df["Formation"].unique()
    results = df["Result"].unique()
    days = df["Day"].unique()
    # For date, get min and max, then sample within range
    date_min = pd.to_datetime(df["Date"], errors="coerce").min()
    date_max = pd.to_datetime(df["Date"], errors="coerce").max()
    # For numeric columns, get mean and std
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    col_stats = {col: (df[col].mean(), df[col].std()) for col in numeric_cols}
    # For categorical columns, get value counts for sampling
    cat_cols = df.select_dtypes(include=[object]).columns
    cat_distributions = {col: df[col].value_counts(normalize=True) for col in cat_cols}

    for i in range(n_rows):
        row = {}
        # Day
        row["Day"] = np.random.choice(days)
        # Match_Day: float, sample from existing
        row["Match_Day"] = float(np.random.choice(df["Match_Day"]))
        # Date: random date in range, add random offset
        rand_days = np.random.randint(0, (date_max - date_min).days)
        row["Date"] = (date_min + timedelta(days=int(rand_days))).strftime(
            "%m/%d/%Y %H:%M"
        )
        # Referee
        row["Referee"] = np.random.choice(referees)
        # Team and Opponent (ensure not the same)
        team = np.random.choice(teams)
        opponent = np.random.choice([t for t in opponents if t != team])
        row["Team"] = team
        row["Opponent"] = opponent
        # Formation
        row["Formation"] = np.random.choice(formations)
        # Possession (0-100)
        row["Possession"] = np.clip(
            np.random.normal(df["Possession"].mean(), df["Possession"].std()), 30, 70
        )
        # Numeric columns
        for col in numeric_cols:
            if col in ["Possession", "Match_Day"]:
                continue
            mean, std = col_stats[col]
            # Some columns are percentages, some are counts, some are floats
            if "Pct" in col or "%" in col or "SoT%" in col:
                val = np.clip(np.random.normal(mean, std), 0, 100)
            elif (
                "Points" in col
                or "Pts" in col
                or "Goals" in col
                or "Conceded" in col
                or "Shots" in col
                or "Fouls" in col
                or "Corners" in col
                or "Yellow" in col
                or "Red" in col
            ):
                val = max(0, int(np.random.normal(mean, std)))
            else:
                val = np.random.normal(mean, std)
            row[col] = val
        # Result
        row["Result"] = np.random.choice(results)
        # Home/Away stats
        for col in df.columns:
            if col.startswith("Home_") or col.startswith("Away_"):
                mean = df[col].mean()
                std = df[col].std()
                row[col] = max(0, int(np.random.normal(mean, std)))
        synthetic_rows.append(row)
    return pd.DataFrame(synthetic_rows, columns=df.columns)


def main():
    # Read original data
    df = pd.read_csv(ORIGINAL_CSV)
    n_rows = 2500
    synthetic_df = generate_synthetic_rows(df, n_rows)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    synthetic_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated {n_rows} synthetic rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
