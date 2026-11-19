
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
EXPLAIN_DIR = REPORT_DIR / "explainability"

RANDOM_STATE = 42
SAMPLE_SIZE = 500


def prepare_directories():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    artifact = joblib.load(
        MODEL_DIR / "churn_final_model.joblib"
    )

    pipeline = artifact["pipeline"]
    features = artifact["features"]
    threshold = float(artifact["threshold"])

    X_train = pd.read_csv(
        DATA_DIR / "X_train.csv"
    )

    missing = set(features) - set(X_train.columns)

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    X_train = X_train[features]

    return pipeline, X_train, threshold


def transform_sample(pipeline, X_train):
    sample = X_train.sample(
        n=min(SAMPLE_SIZE, len(X_train)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    preprocessor = pipeline.named_steps["preprocessor"]

    transformed = preprocessor.transform(sample)

    feature_names = (
        preprocessor.get_feature_names_out().tolist()
    )

    transformed_df = pd.DataFrame(
        np.asarray(transformed),
        columns=feature_names,
    )

    return sample, transformed_df


def calculate_shap(pipeline, transformed_df):
    classifier = pipeline.named_steps["classifier"]

    if classifier.__class__.__name__ != "XGBClassifier":
        raise TypeError(
            "This script expects an XGBoost classifier."
        )

    explainer = shap.TreeExplainer(classifier)

    values = explainer.shap_values(transformed_df)

    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    base_value = float(
        np.asarray(explainer.expected_value).item()
    )

    explanation = shap.Explanation(
        values=values,
        base_values=np.full(
            len(transformed_df),
            base_value,
        ),
        data=transformed_df.to_numpy(),
        feature_names=transformed_df.columns.tolist(),
    )

    # XGBoost binary Tree SHAP generally uses
    # raw-margin/log-odds units.
    # Verify additivity before presenting explanations.
    margins = classifier.predict(
        transformed_df,
        output_margin=True,
    )

    reconstructed = (
        base_value + values.sum(axis=1)
    )

    if not np.allclose(
        reconstructed,
        margins,
        atol=1e-3,
        rtol=1e-3,
    ):
        raise ValueError(
            "SHAP values do not reconstruct model margins."
        )

    return explanation


def save_global_importance(explanation):
    importance = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_absolute_shap": np.abs(
            explanation.values
        ).mean(axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        EXPLAIN_DIR / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTOP MODEL FEATURES")
    print(
        importance.head(15).to_string(index=False)
    )

    return importance


def plot_importance(explanation):
    shap.plots.bar(
        explanation,
        max_display=15,
        show=False,
    )

    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_global_importance.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close("all")


def plot_beeswarm(explanation):
    shap.plots.beeswarm(
        explanation,
        max_display=15,
        show=False,
    )

    plt.title("SHAP Feature Impact Distribution")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_beeswarm.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close("all")


def explain_individual_customers(
    pipeline,
    sample,
    explanation,
    threshold,
):
    probabilities = pipeline.predict_proba(
        sample
    )[:, 1]

    high_index = int(np.argmax(probabilities))
    low_index = int(np.argmin(probabilities))

    selected = {
        "high_risk": high_index,
        "low_risk": low_index,
    }

    records = []

    for label, index in selected.items():
        probability = float(probabilities[index])

        print(f"\n{label.upper()} CUSTOMER")
        print(f"Churn probability: {probability:.2%}")

        shap.plots.waterfall(
            explanation[index],
            max_display=12,
            show=False,
        )

        plt.title(
            f"{label.replace('_', ' ').title()} "
            f"Customer Explanation"
        )

        plt.savefig(
            FIGURE_DIR / f"shap_{label}_customer.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close("all")

        records.append({
            "customer_type": label,
            "predicted_probability": probability,
            "threshold": threshold,
            "predicted_churn": int(
                probability >= threshold
            ),
        })

    pd.DataFrame(records).to_csv(
        EXPLAIN_DIR / "example_explanations.csv",
        index=False,
    )


def create_summary(importance, threshold, n):
    summary = {
        "model": "XGBoost",
        "sample_size": int(n),
        "sample_source": "training_partition",
        "decision_threshold": threshold,
        "shap_output_units": "raw_log_odds",
        "top_features": importance.head(10).to_dict(
            orient="records"
        ),
        "limitations": [
            "SHAP explains model predictions, not causality.",
            "Explanations use a training-data sample.",
            "SHAP values are in raw-margin/log-odds units.",
            "One-hot encoded categories appear separately.",
            "Holdout data was not used for explanations.",
            "Feature availability at prediction time "
            "still requires verification.",
        ],
    }

    with open(
        EXPLAIN_DIR / "explainability_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def main():
    prepare_directories()

    print("\nCUSTOMER CHURN SHAP EXPLAINABILITY")
    print("=" * 60)

    pipeline, X_train, threshold = (
        load_model_and_data()
    )

    print(f"Training records: {len(X_train):,}")
    print(f"Decision threshold: {threshold:.3f}")

    sample, transformed = transform_sample(
        pipeline,
        X_train,
    )

    print(f"SHAP sample size: {len(sample)}")
    print(
        f"Transformed features: {transformed.shape[1]}"
    )

    explanation = calculate_shap(
        pipeline,
        transformed,
    )

    importance = save_global_importance(
        explanation
    )

    print("\nGenerating global importance chart...")
    plot_importance(explanation)

    print("Generating SHAP beeswarm...")
    plot_beeswarm(explanation)

    print("Generating individual explanations...")
    explain_individual_customers(
        pipeline,
        sample,
        explanation,
        threshold,
    )

    create_summary(
        importance,
        threshold,
        len(sample),
    )

    print("\nSHAP ANALYSIS COMPLETE")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Reports: {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
