
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from train_models import (
    build_models,
    make_pipeline,
)
from preprocessing import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(
        DATA_DIR / "y_train.csv"
    )["Exited"].astype(int)

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    )["Exited"].astype(int)

    for X in [X_train, X_test]:
        if set(X.columns) != set(FEATURES):
            raise ValueError("Unexpected model features.")

    if len(X_train) != len(y_train):
        raise ValueError("Training data misaligned.")

    if len(X_test) != len(y_test):
        raise ValueError("Test data misaligned.")

    return (
        X_train[FEATURES],
        X_test[FEATURES],
        y_train,
        y_test,
    )


def select_candidate():
    comparison = pd.read_csv(
        REPORT_DIR / "model_comparison.csv"
    )

    comparison = comparison[
        comparison["model"] != "Dummy"
    ].sort_values(
        "roc_auc_mean",
        ascending=False,
    )

    if comparison.empty:
        raise ValueError("No eligible models found.")

    best_name = comparison.iloc[0]["model"]

    print(f"Selected candidate: {best_name}")

    return best_name


def parameter_space(model_name):
    if model_name == "Logistic Regression":
        return {
            "classifier__C": [
                0.01, 0.1, 1.0, 10.0, 100.0
            ],
            "classifier__class_weight": [
                None, "balanced"
            ],
        }

    if model_name == "Random Forest":
        return {
            "classifier__n_estimators": [
                150, 250, 350
            ],
            "classifier__max_depth": [
                5, 8, 12, None
            ],
            "classifier__min_samples_leaf": [
                1, 3, 5, 10
            ],
            "classifier__max_features": [
                "sqrt", 0.7
            ],
            "classifier__class_weight": [
                None, "balanced_subsample"
            ],
        }

    if model_name == "XGBoost":
        return {
            "classifier__n_estimators": [
                150, 250, 350
            ],
            "classifier__max_depth": [
                2, 3, 4, 5
            ],
            "classifier__learning_rate": [
                0.02, 0.05, 0.1
            ],
            "classifier__subsample": [
                0.75, 0.9, 1.0
            ],
            "classifier__colsample_bytree": [
                0.75, 0.9, 1.0
            ],
            "classifier__reg_lambda": [
                1, 5, 10
            ],
        }

    raise ValueError(f"Unsupported model: {model_name}")


def tune_model(name, X_train, y_train, cv):
    models = build_models()

    pipeline = make_pipeline(models[name])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space(name),
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    print("\nBEST HYPERPARAMETERS")
    print(json.dumps(search.best_params_, indent=2))

    print(
        f"\nBest tuning CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    return search


def optimise_threshold(estimator, X_train, y_train, cv):
    print("\nGenerating out-of-fold probabilities...")

    # Each training observation receives a prediction
    # from a model fitted without that observation.
    # The selected hyperparameters were tuned on the
    # training dataset, so these are not nested-CV
    # estimates of final generalisation performance.
    probabilities = cross_val_predict(
        clone(estimator),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    thresholds = np.linspace(0.05, 0.95, 181)
    records = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        records.append({
            "threshold": float(threshold),
            "precision": precision_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_train,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
        })

    results = pd.DataFrame(records)

    results.to_csv(
        REPORT_DIR / "threshold_analysis.csv",
        index=False,
    )

    # Highest F1; tie-break in favour of lower threshold.
    best = results.sort_values(
        ["f1", "threshold"],
        ascending=[False, True],
    ).iloc[0]

    best_threshold = float(best["threshold"])

    print("\nOPTIMAL TRAINING THRESHOLD")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1-score: {best['f1']:.4f}")

    plt.figure(figsize=(9, 5))

    plt.plot(
        results["threshold"],
        results["precision"],
        label="Precision",
    )
    plt.plot(
        results["threshold"],
        results["recall"],
        label="Recall",
    )
    plt.plot(
        results["threshold"],
        results["f1"],
        label="F1-score",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        color="black",
        label=f"Selected: {best_threshold:.2f}",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Training Threshold Optimisation")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "threshold_optimisation.png",
        dpi=180,
    )
    plt.close()

    return best_threshold


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_confusion(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
    )

    plt.title("Holdout Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=180,
    )
    plt.close()


def plot_roc(y_true, probabilities):
    fpr, tpr, _ = roc_curve(
        y_true, probabilities
    )

    auc = roc_auc_score(
        y_true, probabilities
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.3f}",
    )

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Holdout ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=180,
    )
    plt.close()


def plot_precision_recall(y_true, probabilities):
    precision, recall, _ = precision_recall_curve(
        y_true, probabilities
    )

    ap = average_precision_score(
        y_true, probabilities
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall,
        precision,
        label=f"Average Precision = {ap:.3f}",
    )

    plt.axhline(
        y_true.mean(),
        linestyle="--",
        color="gray",
        label="Churn prevalence",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Holdout Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "precision_recall_curve.png",
        dpi=180,
    )
    plt.close()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    print("\nCUSTOMER CHURN MODEL OPTIMISATION")
    print("=" * 60)

    print(f"Training records: {len(X_train):,}")
    print(f"Holdout records: {len(X_test):,}")

    selected_name = select_candidate()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = tune_model(
        selected_name,
        X_train,
        y_train,
        cv,
    )

    threshold = optimise_threshold(
        search.best_estimator_,
        X_train,
        y_train,
        cv,
    )

    print("\nEVALUATING RESERVED HOLDOUT SET")
    print("=" * 60)

    # Only now do we use the reserved test set.
    # The model is already refitted on the full
    # training partition by RandomizedSearchCV.
    final_model = search.best_estimator_

    probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    default_metrics = calculate_metrics(
        y_test,
        probabilities,
        0.5,
    )

    optimised_metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold,
    )

    print("\nDEFAULT THRESHOLD (0.50)")
    print(json.dumps(default_metrics, indent=2))

    print("\nOPTIMISED THRESHOLD")
    print(json.dumps(optimised_metrics, indent=2))

    plot_confusion(
        y_test,
        probabilities,
        threshold,
    )
    plot_roc(y_test, probabilities)
    plot_precision_recall(
        y_test, probabilities
    )

    # Do not store customer-level holdout predictions
    # in the public reports directory.
    report = {
        "model": selected_name,
        "selection_metric": "training_cv_roc_auc",
        "tuning_cv_roc_auc": float(
            search.best_score_
        ),
        "best_params": search.best_params_,
        "selected_threshold": threshold,
        "default_threshold_metrics": default_metrics,
        "optimised_threshold_metrics": optimised_metrics,
        "training_records": int(len(X_train)),
        "holdout_records": int(len(X_test)),
        "limitations": [
            "Random stratified holdout, not temporal validation.",
            "Model selection and tuning used training data.",
            "Threshold selected using training OOF predictions.",
            "Hyperparameter selection makes OOF scores optimistic.",
            "Complain excluded due to suspected leakage.",
            "Other feature timing remains unverified.",
            "Holdout should not be reused for further tuning.",
        ],
    }

    with open(
        REPORT_DIR / "final_model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    joblib.dump(
        {
            "pipeline": final_model,
            "threshold": threshold,
            "features": FEATURES,
            "model_name": selected_name,
        },
        MODEL_DIR / "churn_final_model.joblib",
    )

    print("\nMODEL EVALUATION COMPLETE")
    print("Final model and reports saved.")
    print("Do not use the holdout for further tuning.")


if __name__ == "__main__":
    main()
