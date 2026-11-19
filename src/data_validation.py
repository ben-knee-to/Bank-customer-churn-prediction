
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
REPORT_DIR = ROOT / "reports"

TARGET = "Exited"

REQUIRED_COLUMNS = [
    "CustomerId",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    print("\nCUSTOMER CHURN DATA VALIDATION")
    print("=" * 55)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing cells: {df.isna().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["CustomerId"].duplicated().any():
        raise ValueError("Duplicate CustomerId values detected.")

    if df[TARGET].isna().any():
        raise ValueError("Missing target values detected.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Exited must contain only 0 and 1.")

    print("\nTARGET DISTRIBUTION")
    print(df[TARGET].value_counts().sort_index())

    churn_rate = df[TARGET].mean()

    print(f"\nOverall churn rate: {churn_rate:.2%}")

    print("\nCHURN BY GEOGRAPHY")
    geography = (
        df.groupby("Geography")[TARGET]
        .agg(["count", "mean"])
        .rename(columns={
            "count": "customers",
            "mean": "churn_rate"
        })
        .sort_values("churn_rate", ascending=False)
    )
    print(geography)

    print("\nCHURN BY ACTIVITY")
    activity = (
        df.groupby("IsActiveMember")[TARGET]
        .agg(["count", "mean"])
    )
    print(activity)

    print("\nCOMPLAINT VS CHURN")
    complaint = pd.crosstab(
        df["Complain"],
        df[TARGET],
        normalize="index"
    )
    print(complaint)

    summary = {
        "total_customers": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_customer_ids": int(df["CustomerId"].nunique()),
        "churned_customers": int(df[TARGET].sum()),
        "retained_customers": int((df[TARGET] == 0).sum()),
        "churn_rate": float(churn_rate),
        "target": TARGET,
        "potential_leakage_columns": ["Complain"],
    }

    with open(
        REPORT_DIR / "data_quality_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    df.describe(include="all").T.to_csv(
        REPORT_DIR / "dataset_profile.csv"
    )

    geography.to_csv(
        REPORT_DIR / "churn_by_geography.csv"
    )

    print("\nValidation reports saved successfully.")


if __name__ == "__main__":
    main()
