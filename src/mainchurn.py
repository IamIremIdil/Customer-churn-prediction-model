import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import kagglehub
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Download dataset
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        df = pd.read_csv(path + '/WA_Fn-UseC_-Telco-Customer-Churn.csv')

        # Clean TotalCharges
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def prepare_ml_data(df):
    """Prepare data for machine learning"""
    df_ml = df.copy()

    # Encode categorical variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        label_encoders[col] = le

    # Encode target
    df_ml['Churn'] = df_ml['Churn'].map({'No': 0, 'Yes': 1})

    return df_ml, label_encoders


@st.cache_data
def train_models(df_ml):
    """Train machine learning models"""
    X = df_ml.drop(['customerID', 'Churn'], axis=1)
    y = df_ml['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    models = {}

    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    models['Logistic Regression'] = {
        'model': lr,
        'predictions': lr_pred,
        'probabilities': lr_proba,
        'auc': roc_auc_score(y_test, lr_proba),
        'X_test': X_test_scaled
    }

    models['Random Forest'] = {
        'model': rf,
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'auc': roc_auc_score(y_test, rf_proba),
        'X_test': X_test
    }

    return models, y_test, X.columns, scaler


def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                                ["📊 Dataset Overview", "📈 Exploratory Analysis", "🤖 ML Models", "🎯 Prediction Tool"])

    # Load data
    with st.spinner("Loading dataset..."):
        df = load_data()

    if df is None:
        st.error("Failed to load dataset. Please check your connection and try again.")
        return

    # Prepare ML data
    df_ml, label_encoders = prepare_ml_data(df)
    models, y_test, feature_names, scaler = train_models(df_ml)

    # Page routing
    if page == "📊 Dataset Overview":
        show_dataset_overview(df)
    elif page == "📈 Exploratory Analysis":
        show_exploratory_analysis(df)
    elif page == "🤖 ML Models":
        show_ml_models(models, y_test, feature_names)
    elif page == "🎯 Prediction Tool":
        show_prediction_tool(df, models, label_encoders, scaler, feature_names)


def show_dataset_overview(df):
    st.header("📊 Dataset Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(df):,}")

    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")

    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

    with col4:
        avg_charges = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_charges:.0f}")

    # Dataset info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Features:**", df.shape[1])
        st.write("**Missing Values:**", df.isnull().sum().sum())

    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes.value_counts())

    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())


def show_exploratory_analysis(df):
    st.header("📈 Exploratory Data Analysis")

    # Churn distribution
    st.subheader("Churn Distribution")

    churn_counts = df['Churn'].value_counts()
    fig = px.pie(values=churn_counts.values, names=churn_counts.index,
                 title="Customer Churn Distribution",
                 color_discrete_sequence=['#ff7f0e', '#1f77b4'])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    # Interactive analysis
    st.subheader("Interactive Analysis")

    analysis_type = st.selectbox("Select Analysis Type",
                                 ["Categorical vs Churn", "Numerical vs Churn", "Correlation Analysis"])

    if analysis_type == "Categorical vs Churn":
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService', 'gender',
                            'Partner', 'Dependents', 'PaperlessBilling']
        selected_col = st.selectbox("Select Categorical Feature", categorical_cols)

        # Create stacked bar chart
        cross_tab = pd.crosstab(df[selected_col], df['Churn'], normalize='index') * 100

        fig = px.bar(cross_tab, title=f'Churn Rate by {selected_col}',
                     labels={'value': 'Percentage', 'index': selected_col})
        st.plotly_chart(fig, use_container_width=True)

        # Show insights
        churn_rates = cross_tab['Yes'].sort_values(ascending=False)
        st.write("**Churn Rates:**")
        for category, rate in churn_rates.items():
            st.write(f"• {category}: {rate:.1f}%")

    elif analysis_type == "Numerical vs Churn":
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        selected_col = st.selectbox("Select Numerical Feature", numerical_cols)

        fig = px.box(df, x='Churn', y=selected_col,
                     title=f'{selected_col} Distribution by Churn Status')
        st.plotly_chart(fig, use_container_width=True)

        # Statistical comparison
        churned = df[df['Churn'] == 'Yes'][selected_col]
        not_churned = df[df['Churn'] == 'No'][selected_col]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churned Customers", f"{churned.mean():.2f}",
                      f"{churned.median():.2f} (median)")
        with col2:
            st.metric("Retained Customers", f"{not_churned.mean():.2f}",
                      f"{not_churned.median():.2f} (median)")

    elif analysis_type == "Correlation Analysis":
        # Correlation matrix
        numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
        numeric_df['Churn_encoded'] = df['Churn'].map({'No': 0, 'Yes': 1})

        corr_matrix = numeric_df.corr()

        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)


def show_ml_models(models, y_test, feature_names):
    st.header("🤖 Machine Learning Models")

    # Model comparison
    st.subheader("Model Performance Comparison")

    model_metrics = []
    for name, result in models.items():
        model_metrics.append({
            'Model': name,
            'AUC Score': f"{result['auc']:.4f}",
            'Accuracy': f"{(result['predictions'] == y_test).mean():.4f}"
        })

    metrics_df = pd.DataFrame(model_metrics)
    st.dataframe(metrics_df)

    # ROC Curves
    st.subheader("ROC Curves")

    fig = go.Figure()

    for name, result in models.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f"{name} (AUC = {result['auc']:.3f})"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Random Classifier', line=dict(dash='dash')))

    fig.update_layout(title='ROC Curves Comparison',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      width=800, height=600)

    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance (Random Forest)
    st.subheader("Feature Importance (Random Forest)")

    rf_model = models['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = px.bar(feature_importance.tail(15), x='Importance', y='Feature',
                 orientation='h', title='Top 15 Most Important Features')
    st.plotly_chart(fig, use_container_width=True)

    # Detailed classification report
    selected_model = st.selectbox("Select model for detailed analysis", list(models.keys()))

    if selected_model:
        result = models[selected_model]

        st.subheader(f"Detailed Analysis: {selected_model}")

        # Classification report
        report = classification_report(y_test, result['predictions'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Confusion matrix
        cm = confusion_matrix(y_test, result['predictions'])
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        title=f"Confusion Matrix - {selected_model}",
                        labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, use_container_width=True)


def show_prediction_tool(df, models, label_encoders, scaler, feature_names):
    st.header("🎯 Customer Churn Prediction Tool")

    st.write("Enter customer information to predict churn probability:")

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", df['gender'].unique())
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", df['Partner'].unique())
        dependents = st.selectbox("Dependents", df['Dependents'].unique())

        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", df['Contract'].unique())
        paperless_billing = st.selectbox("Paperless Billing", df['PaperlessBilling'].unique())
        payment_method = st.selectbox("Payment Method", df['PaymentMethod'].unique())
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", df['PhoneService'].unique())
        multiple_lines = st.selectbox("Multiple Lines", df['MultipleLines'].unique())
        internet_service = st.selectbox("Internet Service", df['InternetService'].unique())
        online_security = st.selectbox("Online Security", df['OnlineSecurity'].unique())
        online_backup = st.selectbox("Online Backup", df['OnlineBackup'].unique())
        device_protection = st.selectbox("Device Protection", df['DeviceProtection'].unique())
        tech_support = st.selectbox("Tech Support", df['TechSupport'].unique())
        streaming_tv = st.selectbox("Streaming TV", df['StreamingTV'].unique())
        streaming_movies = st.selectbox("Streaming Movies", df['StreamingMovies'].unique())

    # Make prediction
    if st.button("Predict Churn", type="primary"):
        # Create input dataframe
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        input_df = pd.DataFrame([input_data])

        # Encode categorical variables
        for col in label_encoders:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Make predictions with both models
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        for i, (name, model_data) in enumerate(models.items()):
            model = model_data['model']

            if name == 'Logistic Regression':
                input_scaled = scaler.transform(input_df)
                probability = model.predict_proba(input_scaled)[0, 1]
            else:
                probability = model.predict_proba(input_df)[0, 1]

            prediction = "High Risk" if probability > 0.5 else "Low Risk"
            color = "red" if probability > 0.5 else "green"

            with col1 if i == 0 else col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <h2 style="color: {color};">{prediction}</h2>
                    <p>Churn Probability: <strong>{probability:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)

        # Risk factors analysis
        st.subheader("Risk Factors Analysis")

        risk_factors = []

        if contract == "Month-to-month":
            risk_factors.append("📅 Month-to-month contract (higher churn risk)")
        if payment_method == "Electronic check":
            risk_factors.append("💳 Electronic check payment (higher churn risk)")
        if internet_service == "Fiber optic":
            risk_factors.append("🌐 Fiber optic internet (mixed risk)")
        if tenure < 12:
            risk_factors.append("⏱️ Low tenure (higher churn risk)")
        if monthly_charges > df['MonthlyCharges'].quantile(0.75):
            risk_factors.append("💰 High monthly charges (higher churn risk)")
        if senior_citizen == 1:
            risk_factors.append("👴 Senior citizen (higher churn risk)")

        protective_factors = []

        if contract in ["One year", "Two year"]:
            protective_factors.append("📋 Long-term contract (lower churn risk)")
        if partner == "Yes":
            protective_factors.append("👥 Has partner (lower churn risk)")
        if dependents == "Yes":
            protective_factors.append("👨‍👩‍👧‍👦 Has dependents (lower churn risk)")
        if tenure > 36:
            protective_factors.append("⭐ High tenure (lower churn risk)")
        if online_security == "Yes":
            protective_factors.append("🔒 Has online security (lower churn risk)")

        if risk_factors:
            st.markdown("**Risk Factors:**")
            for factor in risk_factors:
                st.markdown(f"• {factor}")

        if protective_factors:
            st.markdown("**Protective Factors:**")
            for factor in protective_factors:
                st.markdown(f"• {factor}")

        # Recommendations
        st.subheader("Retention Recommendations")

        recommendations = []

        if probability > 0.7:
            recommendations.extend([
                "🎯 **High Priority**: Immediate intervention needed",
                "📞 Contact customer within 24 hours",
                "💸 Offer retention discount or loyalty program",
                "🔄 Consider contract upgrade incentives"
            ])
        elif probability > 0.4:
            recommendations.extend([
                "⚠️ **Medium Priority**: Monitor closely",
                "📧 Send targeted retention email campaign",
                "🎁 Offer additional services or upgrades",
                "📊 Schedule customer satisfaction survey"
            ])
        else:
            recommendations.extend([
                "✅ **Low Risk**: Standard engagement",
                "💌 Include in general marketing campaigns",
                "🌟 Focus on customer satisfaction maintenance"
            ])

        for rec in recommendations:
            st.markdown(rec)


if __name__ == "__main__":
    main()