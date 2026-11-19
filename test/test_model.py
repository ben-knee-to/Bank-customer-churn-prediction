
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import FEATURES


MODEL_PATH = ROOT / "models/churn_final_model.joblib"
METRICS_PATH = ROOT / "reports/final_model_metrics.json"


@pytest.fixture(scope="module")
def artifact():
    if not MODEL_PATH.exists():
        pytest.skip("Local trained model is unavailable.")

    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_customer():
    return pd.DataFrame([{
        "CreditScore": 650,
        "Age": 42,
        "Tenure": 5,
        "Balance": 75000.0,
        "NumOfProducts": 2,
        "EstimatedSalary": 100000.0,
        "Geography": "France",
        "Gender": "Female",
        "HasCrCard": 1,
        "IsActiveMember": 1,
    }])


def test_feature_count():
    assert len(FEATURES) == 10
    assert len(set(FEATURES)) == len(FEATURES)


def test_no_suspected_leakage_features():
    assert "Complain" not in FEATURES
    assert "Exited" not in FEATURES
    assert "CustomerId" not in FEATURES
    assert "Surname" not in FEATURES


def test_model_artifact(artifact):
    assert "pipeline" in artifact
    assert "threshold" in artifact
    assert "features" in artifact


def test_feature_alignment(artifact, sample_customer):
    assert list(sample_customer.columns) == artifact["features"]


def test_prediction_probability(artifact, sample_customer):
    probability = artifact["pipeline"].predict_proba(
        sample_customer[artifact["features"]]
    )[0, 1]

    assert np.isfinite(probability)
    assert 0 <= probability <= 1


def test_prediction_threshold(artifact):
    assert 0 < float(artifact["threshold"]) < 1


def test_saved_metrics():
    if not METRICS_PATH.exists():
        pytest.skip("Evaluation report is unavailable.")

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    result = metrics["optimised_threshold_metrics"]

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        assert 0 <= result[key] <= 1

    tp = result["true_positives"]
    tn = result["true_negatives"]
    fp = result["false_positives"]
    fn = result["false_negatives"]

    assert tp + tn + fp + fn == metrics["holdout_records"]

    assert np.isclose(
        result["recall"],
        tp / (tp + fn),
    )

    assert np.isclose(
        result["precision"],
        tp / (tp + fp),
    )
