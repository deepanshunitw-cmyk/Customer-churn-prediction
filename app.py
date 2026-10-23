
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

# Load saved model and metrics
@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "rf_model.pkl")

model = load_model()

metrics_path = BASE_DIR / "model_metrics.json"
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
else:
    metrics = {}

st.title("📊 Telco Customer Churn Prediction")
st.write(
    "Predict customer churn using a trained machine learning pipeline "
    "and explore potential retention opportunities."
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Predict Churn", "Model Performance", "About"]
)

# These columns match your trained pipeline
numeric_columns = [
    "MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"
]

categorical_options = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

if page == "Predict Churn":
    st.subheader("Enter Customer Details")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100,
                value=12
            )
            monthly_charges = st.number_input(
                "Monthly Charges", min_value=0.0,
                value=70.0, step=1.0
            )
            total_charges = st.number_input(
                "Total Charges", min_value=0.0,
                value=840.0, step=1.0
            )
            senior_citizen = st.selectbox(
                "Senior Citizen", [0, 1]
            )

        with col2:
            gender = st.selectbox("Gender", categorical_options["gender"])
            partner = st.selectbox("Partner", categorical_options["Partner"])
            dependents = st.selectbox(
                "Dependents", categorical_options["Dependents"]
            )
            phone_service = st.selectbox(
                "Phone Service", categorical_options["PhoneService"]
            )
            multiple_lines = st.selectbox(
                "Multiple Lines", categorical_options["MultipleLines"]
            )

        with col3:
            internet_service = st.selectbox(
                "Internet Service", categorical_options["InternetService"]
            )
            online_security = st.selectbox(
                "Online Security", categorical_options["OnlineSecurity"]
            )
            online_backup = st.selectbox(
                "Online Backup", categorical_options["OnlineBackup"]
            )
            device_protection = st.selectbox(
                "Device Protection", categorical_options["DeviceProtection"]
            )
            tech_support = st.selectbox(
                "Tech Support", categorical_options["TechSupport"]
            )

        col4, col5 = st.columns(2)

        with col4:
            streaming_tv = st.selectbox(
                "Streaming TV", categorical_options["StreamingTV"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", categorical_options["StreamingMovies"]
            )

        with col5:
            contract = st.selectbox(
                "Contract", categorical_options["Contract"]
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                categorical_options["PaperlessBilling"]
            )
            payment_method = st.selectbox(
                "Payment Method", categorical_options["PaymentMethod"]
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        customer = pd.DataFrame([{
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "tenure": tenure,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method
        }])

        prediction = model.predict(customer)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.warning("The model predicts this customer will churn.")
        else:
            st.success("The model predicts this customer will stay.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            classes = list(model.classes_)
            churn_index = classes.index(1)
            churn_probability = probabilities[churn_index]

            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.1%}"
            )
            st.progress(float(churn_probability))

            st.caption(
                "This is a model estimate, not a guarantee of customer behavior."
            )

elif page == "Model Performance":
    st.subheader("Model Evaluation Metrics")

    if metrics:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            with col:
                if isinstance(value, (int, float)):
                    st.metric(name.replace("_", " ").title(), f"{value:.3f}")
                else:
                    st.write(f"**{name}**: {value}")
    else:
        st.info("No evaluation metrics were found in model_metrics.json.")

elif page == "About":
    st.subheader("About This Project")
    st.write("""
    This project predicts customer churn using a trained
    machine learning pipeline.

    **Tools used**
    - Python and Pandas
    - Scikit-learn
    - Streamlit

    **Business objective**

    Help businesses identify customers who may leave and
    support data-driven customer retention decisions.
    """)
