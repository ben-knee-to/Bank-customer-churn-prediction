
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"

REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

TARGET = "Exited"

CATEGORICAL_COLUMNS = [
    "Geography",
    "Gender",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "Satisfaction Score",
    "Card Type",
]

NUMERIC_COLUMNS = [
    "Age",
    "CreditScore",
    "Balance",
    "EstimatedSalary",
    "Point Earned",
]

sns.set_theme(style="whitegrid", palette="deep")


def prepare_directories():
    for directory in [REPORT_DIR, FIGURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError("Missing target column: Exited")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Exited must contain only 0 and 1")

    required = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 65, float("inf")],
        labels=["<=25", "26-35", "36-45",
                "46-55", "56-65", "65+"],
        include_lowest=True,
    )

    df["BalanceGroup"] = pd.cut(
        df["Balance"],
        bins=[-float("inf"), 0, 50000, 100000,
              150000, float("inf")],
        labels=["Zero", "0-50k", "50-100k",
                "100-150k", "150k+"],
    )

    return df


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def churn_summary(df, column):
    result = (
        df.groupby(column, observed=True, dropna=False)[TARGET]
        .agg(
            customers="count",
            churned="sum",
            churn_rate="mean",
        )
        .reset_index()
    )

    result["retained"] = (
        result["customers"] - result["churned"]
    )

    result["churn_rate_pct"] = (
        result["churn_rate"] * 100
    ).round(2)

    result.to_csv(
        TABLE_DIR / f"churn_by_{column.lower()}.csv",
        index=False,
    )

    return result


def plot_churn_distribution(df):
    counts = df[TARGET].value_counts().reindex(
        [0, 1], fill_value=0
    )

    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        x=["Retained", "Churned"],
        y=counts.values,
    )

    for index, value in enumerate(counts.values):
        ax.text(
            index,
            value + 80,
            f"{value:,}",
            ha="center",
            fontsize=11,
        )

    plt.title("Customer Churn Distribution")
    plt.ylabel("Number of Customers")
    plt.xlabel("")
    plt.ylim(0, counts.max() * 1.15)

    save_figure("churn_distribution.png")


def plot_categorical_churn(df, column):
    result = churn_summary(df, column)

    result = result.sort_values(
        "churn_rate_pct", ascending=False
    )

    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        data=result,
        x=column,
        y="churn_rate_pct",
        color="#2878B5",
    )

    ax.set_ylim(
        0,
        max(10, result["churn_rate_pct"].max() * 1.2),
    )

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (
                patch.get_x() + patch.get_width() / 2,
                height,
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(f"Customer Churn by {column}")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel(column)
    plt.xticks(rotation=25)

    save_figure(f"churn_by_{column.lower()}.png")


def plot_numeric_distributions(df):
    for column in NUMERIC_COLUMNS:
        plt.figure(figsize=(9, 5))

        sns.histplot(
            data=df,
            x=column,
            hue=TARGET,
            stat="density",
            common_norm=False,
            bins=30,
            element="step",
            alpha=0.35,
        )

        plt.title(f"{column} Distribution by Churn")
        plt.xlabel(column)
        plt.ylabel("Density")

        save_figure(
            f"distribution_{column.lower()}.png"
        )


def numeric_summary(df):
    result = (
        df.groupby(TARGET)[NUMERIC_COLUMNS]
        .agg(["mean", "median", "std"])
    )

    result.to_csv(
        TABLE_DIR / "numeric_summary_by_churn.csv"
    )

    print("\nNUMERIC FEATURE SUMMARY")
    print(result.round(2))


def correlation_analysis(df):
    # Descriptive analysis only.
    # Exclude identifiers and the suspected leakage column.
    columns = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Satisfaction Score",
        "Point Earned",
        TARGET,
    ]

    correlations = df[columns].corr()

    correlations.to_csv(
        TABLE_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title("Numeric Feature Correlations")

    save_figure("correlation_heatmap.png")


def investigate_complaints(df):
    if "Complain" not in df.columns:
        return

    result = churn_summary(df, "Complain")

    print("\nPOTENTIAL DATA LEAKAGE: COMPLAIN")
    print(result.to_string(index=False))

    print(
        "\nWARNING: Complain is excluded from "
        "predictive modelling pending temporal verification."
    )


def generate_insights(df):
    overall_rate = float(df[TARGET].mean())

    geography = churn_summary(df, "Geography")
    activity = churn_summary(df, "IsActiveMember")
    age = churn_summary(df, "AgeGroup")
    products = churn_summary(df, "NumOfProducts")

    def highest_risk(table, column):
        eligible = table[table["customers"] >= 30]

        if eligible.empty:
            return None

        row = eligible.loc[
            eligible["churn_rate"].idxmax()
        ]

        return {
            "segment": str(row[column]),
            "customers": int(row["customers"]),
            "churn_rate_pct": round(
                float(row["churn_rate"]) * 100, 2
            ),
        }

    insights = {
        "total_customers": int(len(df)),
        "churned_customers": int(df[TARGET].sum()),
        "overall_churn_rate_pct": round(
            overall_rate * 100, 2
        ),
        "highest_risk_geography": highest_risk(
            geography, "Geography"
        ),
        "highest_risk_activity": highest_risk(
            activity, "IsActiveMember"
        ),
        "highest_risk_age_group": highest_risk(
            age, "AgeGroup"
        ),
        "highest_risk_product_count": highest_risk(
            products, "NumOfProducts"
        ),
        "modelling_exclusions": [
            "RowNumber",
            "CustomerId",
            "Surname",
            "Complain",
        ],
        "notes": [
            "Associations do not establish causality.",
            "Segment rankings are descriptive.",
            "Complain may contain target leakage.",
            "Verify feature availability at prediction time.",
        ],
    }

    with open(
        REPORT_DIR / "eda_insights.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(insights, file, indent=4)

    print("\nEDA BUSINESS INSIGHTS")
    print(json.dumps(insights, indent=2))


def main():
    prepare_directories()

    df = load_data()

    print("\nCUSTOMER CHURN EXPLORATORY ANALYSIS")
    print("=" * 55)

    print(f"Customers: {len(df):,}")
    print(f"Churn rate: {df[TARGET].mean():.2%}")

    plot_churn_distribution(df)

    for column in CATEGORICAL_COLUMNS:
        print(f"Analysing: {column}")
        plot_categorical_churn(df, column)

    for column in ["AgeGroup", "BalanceGroup"]:
        print(f"Analysing: {column}")
        plot_categorical_churn(df, column)

    plot_numeric_distributions(df)
    numeric_summary(df)
    correlation_analysis(df)
    investigate_complaints(df)
    generate_insights(df)

    print("\nEDA COMPLETE")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
