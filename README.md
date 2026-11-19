
# Customer Churn Prediction & Analytics

An end-to-end machine learning and customer analytics project
using Python, XGBoost, SHAP and Streamlit.

The project analyses customer churn patterns, compares machine
learning classifiers, optimises churn detection and provides
interactive visualisations and predictions.

## Project Overview

Customer churn is an important challenge for financial institutions.

This project uses historical bank customer records to:

- Explore customer churn patterns.
- Identify customer segments associated with churn.
- Train and compare classification models.
- Optimise classification thresholds.
- Explain predictions using SHAP.
- Build an interactive Streamlit dashboard.

## Dataset

Source: Customer-Churn-Records.csv

Records: 10,000
Original columns: 18
Target: Exited

Churned customers: 2,038
Retained customers: 7,962
Overall churn rate: 20.38%

The dataset includes customer demographics, account information,
product ownership, activity status and churn outcomes.

The Complain feature was excluded because it is almost perfectly
associated with the target and presents a serious leakage risk.

Customer identifiers and other potentially unsuitable features
were also excluded from modelling.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Pytest
- Git and GitHub

## Exploratory Data Analysis

Key findings:

| Segment | Churn Rate |
|---|---:|
| Overall | 20.38% |
| Germany | 32.44% |
| France | 16.17% |
| Spain | 16.67% |
| Inactive customers | 26.87% |
| Active customers | 14.27% |
| Ages 46–55 | 50.57% |

These findings describe historical associations, not causal effects.

### 1. What is the overall churn rate?
**Finding:** 2,038 of 10,000 customers churned (20.38%).
**Recommendation:** Monitor churn trends and estimate customer retention value.

### 2. Which geography has the highest churn?
**Finding:** Germany 32.44%, Spain 16.67%, France 16.17%.
**Recommendation:** Investigate regional differences in customer experience and product fit.

### 3. Does activity relate to churn?
**Finding:** Inactive customers churned at 26.87%, versus 14.27% for active customers.
**Recommendation:** Test engagement outreach with a controlled experiment.

### 4. Which age segment has elevated churn?
**Finding:** Ages 46–55 had 50.57% churn among 1,311 customers.
**Recommendation:** Investigate service needs and review fairness before operational use.

### 5. How does product ownership relate to churn?
**Finding:** All 60 four-product customers were recorded as churned.
**Recommendation:** Check this small segment and potential data artefacts before generalising.

### 6. Which features most influence the XGBoost model?
**Finding:** Mean absolute SHAP values: Age 0.8618, NumOfProducts 0.8064,
IsActiveMember_0 0.3210, Geography_Germany 0.2326, Balance 0.2248.
**Recommendation:** Review feature quality, stability and fairness. SHAP does not establish causation.

### 7. Can machine learning support targeted retention analysis?
**Finding:** Reserved holdout ROC-AUC 0.8759; average precision 0.7249.
At threshold 0.31, precision 61.09%, recall 66.18%, F1 0.6353.
The model identified 270 of 408 churners and incorrectly flagged 172 retained customers.
**Recommendation:** Evaluate intervention costs and benefits in a controlled pilot before deployment.

### Business conclusion
Historical churn differs by segment, and XGBoost demonstrates useful predictive discrimination.
These observations do not establish causal effects, financial savings or future deployment performance.

## Machine Learning

Models compared:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest
4. XGBoost

XGBoost was selected based on training-only cross-validation.

Hyperparameter tuning used RandomizedSearchCV with five-fold
stratified cross-validation.

Best tuning cross-validation ROC-AUC: 0.8640

Selected hyperparameters:

- n_estimators: 350
- max_depth: 3
- learning_rate: 0.05
- subsample: 0.9
- colsample_bytree: 0.9
- reg_lambda: 1

## Holdout Evaluation

The reserved holdout dataset contained 2,000 customers.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8759 |
| Average Precision | 0.7249 |
| Accuracy | 84.50% |
| Precision | 61.09% |
| Recall | 66.18% |
| F1-score | 0.6353 |
| Decision threshold | 0.31 |

Confusion matrix:

| | Predicted Retained | Predicted Churn |
|---|---:|---:|
| Actual Retained | 1,420 | 172 |
| Actual Churn | 138 | 270 |

The selected threshold improved recall relative to the default
0.50 threshold, while increasing false positives.

Threshold selection used training-only out-of-fold predictions.

The holdout dataset was not used for hyperparameter or
threshold selection.

## Explainable AI

SHAP was used to interpret XGBoost predictions.

Top features by mean absolute SHAP value:

| Feature | Importance |
|---|---:|
| Age | 0.8618 |
| NumOfProducts | 0.8064 |
| IsActiveMember_0 | 0.3210 |
| Geography_Germany | 0.2326 |
| Balance | 0.2248 |
| Gender_Female | 0.2155 |

SHAP explanations describe model behaviour, not causal effects.

## Streamlit Dashboard

The application contains five pages:

1. Executive Overview
2. Customer Segmentation
3. Churn Prediction
4. Explainable AI
5. Model Performance

## Installation

Clone the repository:

```powershell
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

Create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The original dataset and trained model are not included by default.

To reproduce the project, obtain the dataset from its authorised
source and place it at:

data/raw/Customer-Churn-Records.csv

Run the scripts in this order:

```powershell
python src/data_validation.py
python src/churn_analysis.py
python "src/data preparation.py"
python src/train_models.py
python src/model_evaluation.py
python src/model_explainability.py
```

Note: Running model_evaluation.py repeats tuning and holdout
evaluation. Use this only for a fresh reproducibility run,
not for selecting changes based on previously observed
holdout results.

Launch the dashboard:

```powershell
python -m streamlit run dashboard.py
```

The prediction page requires the locally generated model artifact.

## Testing

```powershell
python -m pytest tests -v
```

## Limitations

- Historical associations do not establish causation.
- The data may not represent current banking customers.
- The split is random rather than time-based.
- Feature availability at prediction time is not fully verified.
- Probabilities have not been independently calibrated.
- The selected threshold is F1-based, not profit-optimised.
- Gender and geography require fairness review before operational use.
- The model is a portfolio demonstration, not a production banking system.

## Future Improvements

- Temporal validation on new customer data.
- Probability calibration and drift monitoring.
- Fairness and subgroup performance assessment.
- Business cost-sensitive threshold optimisation.
- Model deployment and automated monitoring.

