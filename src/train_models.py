
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from preprocessing import build_preprocessor, FEATURES


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_SPLITS = 5
DEFAULT_THRESHOLD = 0.50


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    X_train = pd.read_csv(
        DATA_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        DATA_DIR / "y_train.csv"
    )["Exited"].astype(int)

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training features and labels do not align."
        )

    if X_train.isna().all(axis=0).any():
        raise ValueError(
            "A training feature is entirely missing."
        )

    missing = set(FEATURES) - set(X_train.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    if set(X_train.columns) != set(FEATURES):
        raise ValueError(
            "Unexpected columns in training data. "
            "Re-run data preparation."
        )

    if not y_train.isin([0, 1]).all():
        raise ValueError(
            "Target must be binary."
        )

    if y_train.nunique() != 2:
        raise ValueError(
            "Training target must contain both classes."
        )

    if y_train.value_counts().min() < N_SPLITS:
        raise ValueError(
            "Each target class must have at least "
            f"{N_SPLITS} records for stratified CV."
        )

    return X_train[FEATURES], y_train


# ============================================================
# BUILD MACHINE LEARNING MODELS
# ============================================================

def build_models():

    return {

        "Dummy": DummyClassifier(
            strategy="prior"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


# ============================================================
# BUILD PIPELINE
# ============================================================

def make_pipeline(model):

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model),
    ])


# ============================================================
# CROSS-VALIDATION SCORING
# ============================================================

def build_scoring():

    return {
        "roc_auc": "roc_auc",

        "average_precision": "average_precision",

        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),

        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),

        "accuracy": "accuracy",
    }


# ============================================================
# EVALUATE EACH MODEL
# ============================================================

def evaluate_model(name, model, X, y, cv):

    print(f"\nTraining: {name}")
    print("-" * 55)

    pipeline = make_pipeline(model)

    start = time.perf_counter()

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=build_scoring(),
        n_jobs=1,
        error_score="raise",
    )

    elapsed = time.perf_counter() - start

    result = {
        "model": name,

        "roc_auc_mean": float(
            scores["test_roc_auc"].mean()
        ),

        "roc_auc_std": float(
            scores["test_roc_auc"].std()
        ),

        "pr_auc_mean": float(
            scores["test_average_precision"].mean()
        ),

        "f1_mean": float(
            scores["test_f1"].mean()
        ),

        "precision_mean": float(
            scores["test_precision"].mean()
        ),

        "recall_mean": float(
            scores["test_recall"].mean()
        ),

        "accuracy_mean": float(
            scores["test_accuracy"].mean()
        ),

        "cv_seconds": round(elapsed, 2),
    }

    for metric in [
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "accuracy_mean",
    ]:

        print(
            f"{metric}: {result[metric]:.4f}"
        )

    print(f"CV time: {elapsed:.2f} seconds")

    return result


# ============================================================
# GENERATE OUT-OF-FOLD PREDICTIONS
# ============================================================

def generate_oof_predictions(name, model, X, y, cv):

    print(
        f"\nGenerating out-of-fold predictions: {name}"
    )

    pipeline = make_pipeline(model)

    probabilities = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    predicted_churn_count = int(
        predictions.sum()
    )

    predicted_churn_rate = float(
        predictions.mean()
    )

    actual_churn_count = int(
        y.sum()
    )

    true_positives = int(
        ((y.to_numpy() == 1) & (predictions == 1)).sum()
    )

    false_positives = int(
        ((y.to_numpy() == 0) & (predictions == 1)).sum()
    )

    false_negatives = int(
        ((y.to_numpy() == 1) & (predictions == 0)).sum()
    )

    true_negatives = int(
        ((y.to_numpy() == 0) & (predictions == 0)).sum()
    )

    metrics = {
        "model": name,

        "threshold": DEFAULT_THRESHOLD,

        "roc_auc": float(
            roc_auc_score(y, probabilities)
        ),

        "pr_auc": float(
            average_precision_score(y, probabilities)
        ),

        "precision_0_5": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "recall_0_5": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "f1_0_5": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),

        "actual_churn_count": actual_churn_count,

        "predicted_churn_count": predicted_churn_count,

        "predicted_churn_rate": predicted_churn_rate,

        "true_positives": true_positives,

        "false_positives": false_positives,

        "false_negatives": false_negatives,

        "true_negatives": true_negatives,
    }

    print(
        f"Actual churners: {actual_churn_count:,}"
    )

    print(
        f"Predicted churners: {predicted_churn_count:,}"
    )

    print(
        f"Predicted churn rate: {predicted_churn_rate:.2%}"
    )

    if predicted_churn_count == 0:
        print(
            "WARNING: This model predicted no churners "
            "at the default threshold."
        )

    print(
        f"OOF ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    print(
        f"OOF PR-AUC: {metrics['pr_auc']:.4f}"
    )

    print(
        f"OOF Precision: {metrics['precision_0_5']:.4f}"
    )

    print(
        f"OOF Recall: {metrics['recall_0_5']:.4f}"
    )

    print(
        f"OOF F1: {metrics['f1_0_5']:.4f}"
    )

    return probabilities, predictions, metrics


# ============================================================
# SAVE SELECTED MODEL OOF PREDICTIONS
# ============================================================

def save_oof_predictions(name, model, X, y, cv):

    probabilities, predictions, metrics = (
        generate_oof_predictions(
            name,
            model,
            X,
            y,
            cv,
        )
    )

    output = pd.DataFrame({
        "actual": y.to_numpy(),
        "churn_probability": probabilities,
        "prediction_0_5": predictions,
    })

    output.to_csv(
        REPORT_DIR / "training_oof_predictions.csv",
        index=False,
    )

    return metrics


# ============================================================
# MAIN TRAINING WORKFLOW
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train, y_train = load_training_data()

    print("\nCUSTOMER CHURN MODEL TRAINING")
    print("=" * 55)

    print(
        f"Training records: {len(X_train):,}"
    )

    print(
        f"Churn rate: {y_train.mean():.2%}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    results = []

    # --------------------------------------------------------
    # STEP 1: COMPARE MODELS
    # --------------------------------------------------------

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            cv,
        )

        results.append(result)

    comparison = pd.DataFrame(
        results
    ).sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print("=" * 55)

    print(
        comparison[
            [
                "model",
                "roc_auc_mean",
                "pr_auc_mean",
                "precision_mean",
                "recall_mean",
                "f1_mean",
            ]
        ].round(4).to_string(index=False)
    )

    # --------------------------------------------------------
    # STEP 2: SELECT BEST NON-DUMMY MODEL
    # --------------------------------------------------------

    eligible = comparison[
        comparison["model"] != "Dummy"
    ]

    best_name = eligible.iloc[0]["model"]

    best_model = models[best_name]

    print(
        f"\nBest candidate: {best_name}"
    )

    # --------------------------------------------------------
    # STEP 3: GENERATE SELECTED MODEL OOF PREDICTIONS
    # --------------------------------------------------------

    oof_metrics = save_oof_predictions(
        best_name,
        best_model,
        X_train,
        y_train,
        cv,
    )

    print("\nOUT-OF-FOLD METRICS")
    print(
        json.dumps(
            oof_metrics,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # STEP 4: FIT SELECTED MODEL ON ALL TRAINING DATA
    # --------------------------------------------------------

    print("\nFitting selected candidate model...")

    final_pipeline = make_pipeline(
        best_model
    )

    final_pipeline.fit(
        X_train,
        y_train,
    )

    model_path = (
        MODEL_DIR / "churn_candidate.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_path,
    )

    # --------------------------------------------------------
    # STEP 5: SAVE MODEL METADATA
    # --------------------------------------------------------

    metadata = {
        "selected_model": best_name,

        "selection_metric": "mean_5_fold_roc_auc",

        "training_records": int(
            len(X_train)
        ),

        "training_churn_rate": float(
            y_train.mean()
        ),

        "features": FEATURES,

        "random_state": RANDOM_STATE,

        "cross_validation_folds": N_SPLITS,

        "default_threshold": DEFAULT_THRESHOLD,

        "holdout_evaluated": False,

        "oof_metrics": oof_metrics,

        "notes": [
            "All preprocessing fitted inside CV folds.",
            "Complain excluded due to potential leakage.",
            "Holdout test set not used for model selection.",
            "Undefined precision handled with zero_division=0.",
            "OOF predictions generated on training data only.",
            "Candidate is not yet approved for deployment.",
        ],
    }

    with open(
        REPORT_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # STEP 6: COMPLETION SUMMARY
    # --------------------------------------------------------

    print("\nTRAINING COMPLETE")
    print("=" * 55)

    print(
        f"Candidate saved: {model_path}"
    )

    print(
        "Comparison saved: "
        f"{REPORT_DIR / 'model_comparison.csv'}"
    )

    print(
        "OOF predictions saved: "
        f"{REPORT_DIR / 'training_oof_predictions.csv'}"
    )

    print(
        "Metadata saved: "
        f"{REPORT_DIR / 'model_metadata.json'}"
    )

    print(
        "\nHoldout test set remains untouched."
    )


if __name__ == "__main__":
    main()
