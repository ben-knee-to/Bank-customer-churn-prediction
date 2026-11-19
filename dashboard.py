from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'raw' / 'Customer-Churn-Records.csv'
MODEL_PATH = ROOT / 'models' / 'churn_final_model.joblib'
METRICS_PATH = ROOT / 'reports' / 'final_model_metrics.json'
SHAP_PATH = ROOT / 'reports' / 'explainability' / 'shap_feature_importance.csv'
FIGURES = ROOT / 'reports' / 'figures'

st.set_page_config(page_title='Customer Churn Intelligence', page_icon='📊', layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_shap():
    return pd.read_csv(SHAP_PATH) if SHAP_PATH.exists() else pd.DataFrame()

@st.cache_resource
def load_model():
    # Only load trusted, locally created joblib files.
    return joblib.load(MODEL_PATH)

def churn_by(df, column):
    return df.groupby(column, observed=True)['Exited'].agg(
        customers='count', churned='sum', churn_rate='mean'
    ).reset_index()

def churn_chart(df, column, title):
    summary = churn_by(df, column)
    summary['churn_pct'] = 100 * summary['churn_rate']
    summary[column] = summary[column].astype(str)
    fig = px.bar(summary, x=column, y='churn_pct', text='churn_pct',
                 title=title, hover_data=['customers', 'churned'])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(yaxis_title='Churn rate (%)', xaxis_title='',
                      yaxis_range=[0, max(10, summary['churn_pct'].max() * 1.2)])
    return fig

def show_saved_figure(filename, title):
    path = FIGURES / filename
    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)

def overview(df, metrics):
    st.title('Customer Churn Intelligence')
    st.caption('Historical customer analytics and predictive modelling')
    result = metrics['optimised_threshold_metrics']
    a, b, c, d = st.columns(4)
    a.metric('Customers', f'{len(df):,}')
    b.metric('Churned', f'{int(df.Exited.sum()):,}')
    c.metric('Churn rate', f'{df.Exited.mean():.2%}')
    d.metric('Holdout ROC-AUC', f'{result["roc_auc"]:.4f}')
    left, right = st.columns(2)
    with left:
        counts = pd.DataFrame({'Status': ['Retained', 'Churned'],
                               'Customers': [int((df.Exited == 0).sum()), int(df.Exited.sum())]})
        st.plotly_chart(px.pie(counts, names='Status', values='Customers',
                               hole=0.55, title='Customer outcomes'), use_container_width=True)
    with right:
        st.plotly_chart(churn_chart(df, 'Geography', 'Churn by geography'),
                        use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(churn_chart(df, 'IsActiveMember', 'Churn by activity'),
                        use_container_width=True)
    with right:
        st.plotly_chart(churn_chart(df, 'NumOfProducts', 'Churn by product count'),
                        use_container_width=True)
    st.caption('Associations do not establish causality.')

def segmentation(df):
    st.title('Customer Segmentation')
    countries = st.multiselect('Geography', sorted(df.Geography.dropna().unique()),
                               default=sorted(df.Geography.dropna().unique()))
    genders = st.multiselect('Gender', sorted(df.Gender.dropna().unique()),
                             default=sorted(df.Gender.dropna().unique()))
    ages = st.slider('Age', int(df.Age.min()), int(df.Age.max()),
                     (int(df.Age.min()), int(df.Age.max())))
    if not countries or not genders:
        st.warning('Select at least one geography and gender.')
        return
    selected = df[df.Geography.isin(countries) & df.Gender.isin(genders)
                  & df.Age.between(*ages)].copy()
    if selected.empty:
        st.warning('No records match your filters.')
        return
    a, b, c = st.columns(3)
    a.metric('Customers', f'{len(selected):,}')
    b.metric('Churn rate', f'{selected.Exited.mean():.2%}')
    c.metric('Average age', f'{selected.Age.mean():.1f}')
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(selected, x='Age', color='Exited', nbins=30,
                                    title='Age distribution by outcome'), use_container_width=True)
        st.plotly_chart(churn_chart(selected, 'Geography', 'Churn by geography'),
                        use_container_width=True)
    with right:
        st.plotly_chart(px.box(selected, x='Exited', y='Balance',
                               title='Balance by churn outcome'), use_container_width=True)
        st.plotly_chart(churn_chart(selected, 'NumOfProducts', 'Churn by products'),
                        use_container_width=True)
    summary = churn_by(selected, 'Geography')
    summary['churn_rate'] = (100 * summary['churn_rate']).round(2)
    st.dataframe(summary.rename(columns={'churn_rate': 'churn_rate_pct'}),
                 hide_index=True, use_container_width=True)
    st.caption('Small segments can have unstable rates.')

def prediction(artifact):
    st.title('Customer Churn Prediction')
    st.caption('Portfolio demonstration; not validated for operational banking decisions.')
    pipeline = artifact['pipeline']
    features = artifact['features']
    threshold = float(artifact['threshold'])
    with st.form('customer_form'):
        left, right = st.columns(2)
        with left:
            credit = st.number_input('Credit score', 300, 900, 650)
            age = st.number_input('Age', 18, 100, 40)
            tenure = st.slider('Tenure (years)', 0, 10, 5)
            balance = st.number_input('Balance', min_value=0.0, value=75000.0, step=1000.0)
            products = st.selectbox('Number of products', [1, 2, 3, 4])
        with right:
            salary = st.number_input('Estimated salary', min_value=0.0,
                                     value=100000.0, step=1000.0)
            geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
            gender = st.selectbox('Gender', ['Female', 'Male'])
            card = st.selectbox('Has credit card', ['Yes', 'No'])
            active = st.selectbox('Active member', ['Yes', 'No'])
        submitted = st.form_submit_button('Predict churn risk', type='primary')
    if not submitted:
        return
    customer = pd.DataFrame([{
        'CreditScore': credit, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': products,
        'EstimatedSalary': salary, 'Geography': geography,
        'Gender': gender, 'HasCrCard': int(card == 'Yes'),
        'IsActiveMember': int(active == 'Yes')
    }])[features]
    probability = float(pipeline.predict_proba(customer)[0, 1])
    a, b = st.columns(2)
    a.metric('Estimated churn probability', f'{probability:.2%}')
    b.metric('Selected decision threshold', f'{threshold:.2f}')
    if probability >= threshold:
        st.warning('Higher predicted churn risk at the selected threshold.')
    else:
        st.success('Lower predicted churn risk at the selected threshold.')
    gauge = go.Figure(go.Indicator(mode='gauge+number', value=100 * probability,
        number={'suffix': '%'}, gauge={'axis': {'range': [0, 100]},
        'threshold': {'line': {'color': 'red', 'width': 3}, 'value': 100 * threshold}}))
    st.plotly_chart(gauge, use_container_width=True)
    st.caption('Predicted probabilities may not be perfectly calibrated.')

def explainability(shap_df):
    st.title('Explainable AI — SHAP')
    if shap_df.empty:
        st.warning('SHAP report missing; run model_explainability.py first.')
        return
    top = shap_df.sort_values('mean_absolute_shap').tail(15)
    fig = px.bar(top, x='mean_absolute_shap', y='feature', orientation='h',
                 title='Global feature importance (mean absolute SHAP)')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(shap_df.sort_values('mean_absolute_shap', ascending=False),
                 hide_index=True, use_container_width=True)
    show_saved_figure('shap_beeswarm.png', 'SHAP impact distribution')
    show_saved_figure('shap_high_risk_customer.png', 'Example: high predicted risk')
    show_saved_figure('shap_low_risk_customer.png', 'Example: low predicted risk')
    st.info('SHAP explains model predictions, not causality. These SHAP values are in raw log-odds.')

def performance(metrics):
    st.title('Model Performance')
    default = metrics['default_threshold_metrics']
    chosen = metrics['optimised_threshold_metrics']
    a, b, c, d = st.columns(4)
    a.metric('Holdout ROC-AUC', f'{chosen["roc_auc"]:.4f}')
    b.metric('Average precision', f'{chosen["pr_auc"]:.4f}')
    c.metric('Recall', f'{chosen["recall"]:.2%}')
    d.metric('F1', f'{chosen["f1"]:.4f}')
    comparison = pd.DataFrame([{'Threshold': label, **{
        k: values[k] for k in ('accuracy', 'precision', 'recall', 'f1')}}
        for label, values in [('0.50', default), ('0.31', chosen)]])
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    matrix = np.array([[chosen['true_negatives'], chosen['false_positives']],
                       [chosen['false_negatives'], chosen['true_positives']]])
    fig = px.imshow(matrix, x=['Predicted retained', 'Predicted churn'],
                    y=['Actual retained', 'Actual churn'], text_auto=True,
                    title='Holdout confusion matrix', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    for filename, title in [('roc_curve.png', 'ROC curve'),
                            ('precision_recall_curve.png', 'Precision–recall curve'),
                            ('threshold_optimisation.png', 'Training threshold analysis')]:
        show_saved_figure(filename, title)
    st.warning('Do not use the reserved holdout set for additional model or threshold tuning.')

def business_insights(df, metrics, shap_df):
    st.title('Business Insights & Recommendations')
    st.caption('Seven questions answered from historical observations and holdout evaluation.')
    result = metrics['optimised_threshold_metrics']
    a, b, c, d = st.columns(4)
    a.metric('Customers', f'{len(df):,}')
    b.metric('Churned', f'{int(df.Exited.sum()):,}')
    c.metric('Churn rate', f'{df.Exited.mean():.2%}')
    d.metric('Holdout ROC-AUC', f'{result["roc_auc"]:.4f}')

    st.subheader('1. What is the overall churn rate?')
    st.write(f'**{int(df.Exited.sum()):,} of {len(df):,} customers churned ({df.Exited.mean():.2%}).**')
    st.info('Recommendation: Track churn over time and estimate the value of retaining customers.')

    st.subheader('2. Which geography has the highest churn?')
    st.plotly_chart(churn_chart(df, 'Geography', 'Churn by geography'), use_container_width=True)
    geography = churn_by(df, 'Geography').sort_values('churn_rate', ascending=False)
    top = geography.iloc[0]
    st.write(f'**Finding:** {top.Geography} had the highest observed churn ({top.churn_rate:.2%}).')
    st.info('Recommendation: Investigate service, product fit and customer experience across regions.')

    st.subheader('3. Does customer activity relate to churn?')
    st.plotly_chart(churn_chart(df, 'IsActiveMember', 'Churn by activity'), use_container_width=True)
    activity = churn_by(df, 'IsActiveMember').set_index('IsActiveMember')
    if 0 in activity.index and 1 in activity.index:
        st.write(f'**Finding:** Inactive: {activity.loc[0, "churn_rate"]:.2%}; '
                 f'active: {activity.loc[1, "churn_rate"]:.2%}.')
    st.info('Recommendation: Test engagement outreach in a controlled retention pilot.')

    st.subheader('4. Which age segment has elevated churn?')
    age_group = df[df.Age.between(46, 55)]
    if len(age_group):
        st.write(f'**Finding:** Ages 46–55: {age_group.Exited.mean():.2%} churn '
                 f'over {len(age_group):,} records.')
    st.plotly_chart(px.histogram(df, x='Age', color='Exited', nbins=30,
                                 title='Age distribution by churn'), use_container_width=True)
    st.info('Recommendation: Investigate service needs, with fairness review before age-based decisions.')

    st.subheader('5. How does product ownership relate to churn?')
    st.plotly_chart(churn_chart(df, 'NumOfProducts', 'Churn by number of products'),
                    use_container_width=True)
    products = churn_by(df, 'NumOfProducts')
    four = products[products.NumOfProducts == 4]
    if not four.empty:
        item = four.iloc[0]
        st.write(f'**Finding:** Four-product customers: {item.churn_rate:.2%} churn '
                 f'over {int(item.customers)} records.')
    st.info('Recommendation: Investigate this small segment and possible data artefacts.')

    st.subheader('6. What influences the model predictions?')
    if shap_df.empty:
        st.warning('SHAP report unavailable.')
    else:
        top = shap_df.nlargest(8, 'mean_absolute_shap').sort_values('mean_absolute_shap')
        st.plotly_chart(px.bar(top, x='mean_absolute_shap', y='feature', orientation='h',
                               title='Top SHAP features'), use_container_width=True)
    st.info('Recommendation: Review feature stability, data quality and fairness. SHAP is not causal.')

    st.subheader('7. Can machine learning identify potential churners?')
    a, b, c = st.columns(3)
    a.metric('Precision', f'{result["precision"]:.2%}')
    b.metric('Recall', f'{result["recall"]:.2%}')
    c.metric('F1-score', f'{result["f1"]:.4f}')
    tp, fn, fp = (int(result[k]) for k in
                  ('true_positives', 'false_negatives', 'false_positives'))
    st.write(f'**Finding:** The model identified {tp} of {tp + fn} churners '
             f'and incorrectly flagged {fp} retained customers.')
    st.info('Recommendation: Pilot risk-informed outreach and measure incremental retention benefit and cost.')
    st.divider()
    st.success('Conclusion: Churn varies by segment, and the model has useful holdout discrimination.')
    st.caption('No causal benefit, financial savings or live deployment performance has been established.')

def main():
    try:
        df = load_data()
        metrics = load_metrics()
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Required input unavailable: {exc}')
        st.stop()
    st.sidebar.title('Customer Intelligence')
    page = st.sidebar.radio('Navigation', [
        'Executive Overview', 'Business Insights', 'Customer Segmentation',
        'Churn Prediction', 'Explainable AI', 'Model Performance'
    ])
    st.sidebar.caption('Model: XGBoost | Portfolio demonstration')
    if page == 'Executive Overview':
        overview(df, metrics)
    elif page == 'Business Insights':
        business_insights(df, metrics, load_shap())
    elif page == 'Customer Segmentation':
        segmentation(df)
    elif page == 'Churn Prediction':
        try:
            prediction(load_model())
        except (OSError, ValueError, KeyError) as exc:
            st.error(f'Model unavailable: {exc}')
    elif page == 'Explainable AI':
        explainability(load_shap())
    elif page == 'Model Performance':
        performance(metrics)

if __name__ == '__main__':
    main()
