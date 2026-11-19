
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Exited"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "HasCrCard",
    "IsActiveMember",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXCLUDED_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname",
    "Complain",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    return df


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    print("\nCUSTOMER CHURN DATA PREPARATION")
    print("=" * 55)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Reset indexes so saved feature and target files align.
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Training records: {len(X_train):,}")
    print(f"Test records: {len(X_test):,}")
    print(f"Training churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    # Save the raw feature partitions.
    # Preprocessing will be fitted inside model pipelines.
    X_train.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False,
    )

    X_test.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False,
    )

    y_train.to_frame(TARGET).to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_frame(TARGET).to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
    )

    # Diagnostic preprocessing only.
    # Do not use this fitted object in cross-validation.
    # Each model will fit its own preprocessing pipeline.
    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    print("\nFEATURE ENGINEERING")
    print(f"Original features: {X_train.shape[1]}")
    print(
        f"Transformed features: "
        f"{X_train_transformed.shape[1]}"
    )

    print("\nTRANSFORMED FEATURE NAMES")
    for feature in feature_names:
        print(f" - {feature}")

    assert np.isfinite(X_train_transformed).all()
    assert np.isfinite(X_test_transformed).all()

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

    # Save a diagnostic preprocessing artifact.
    # Model training must use fresh, unfitted pipelines.
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor_diagnostic.joblib",
    )

    summary = {
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "training_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "original_feature_count": int(X.shape[1]),
        "transformed_feature_count": int(
            X_train_transformed.shape[1]
        ),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": EXCLUDED_COLUMNS,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "split_method": "stratified_random",
        "feature_names": feature_names,
        "notes": [
            "Holdout test set reserved for final evaluation.",
            "Preprocessing must be refitted within each CV fold.",
            "Complain excluded due to suspected target leakage.",
            "Feature timing has not been independently verified.",
        ],
    }

    with open(
        REPORT_DIR / "preprocessing_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nDATA PREPARATION COMPLETE")
    print("All validation checks passed.")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Diagnostic preprocessor: {MODEL_DIR}")
    print(f"Summary report: {REPORT_DIR}")


if __name__ == "__main__":
    main()
