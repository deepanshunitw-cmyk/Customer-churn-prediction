# 📊 Telco Customer Churn Prediction

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. When customers discontinue a service, businesses may lose recurring revenue and incur additional costs to acquire new customers.

This project uses Machine Learning to predict whether a telecommunications customer is likely to churn based on customer demographics, account information, subscribed services, contract details, and billing information.

The objective is to help businesses identify customers who may be at risk of leaving, understand churn-related patterns, and support data-driven customer retention strategies.

## 🎯 Project Objectives

- Predict customer churn using a supervised Machine Learning model.
- Analyze customer demographics, service usage, tenure, and billing information.
- Build a reusable prediction pipeline with Scikit-learn.
- Evaluate model performance using appropriate classification metrics.
- Develop an interactive Streamlit application for customer-level predictions.
- Demonstrate how churn predictions can support business decision-making.

## 🗂️ Dataset Description

The project uses the Telco Customer Churn dataset, which contains information about telecommunications customers and whether they discontinued their service.

The model uses the following **19 input features**:

| Category | Features |
|---|---|
| Customer Information | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account Information | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Billing Information | `MonthlyCharges`, `TotalCharges` |
| Phone Services | `PhoneService`, `MultipleLines` |
| Internet Services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

**Target variable:** `Churn`

The target represents whether a customer churned. The exact encoding of the target classes should be verified against the training notebook before interpreting class `0` or class `1`.

## ⚙️ Technologies Used

- **Python** — Core programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Scikit-learn** — Machine Learning pipeline and model evaluation
- **Joblib** — Saving and loading the trained model
- **Streamlit** — Interactive web application
- **Git and GitHub** — Version control and project hosting
- **Streamlit Community Cloud** — Web application deployment

## 🧠 Machine Learning Approach

### 1. Data Preparation

The customer dataset is prepared for Machine Learning by examining the data, handling relevant data-quality issues, and organizing the input features and target variable.

### 2. Feature Processing

The model uses numerical and categorical features. Preprocessing is integrated into the saved Scikit-learn pipeline, allowing the same transformations to be applied during prediction.

### 3. Model Training

A Random Forest classifier is used to predict customer churn. The trained pipeline is saved as `rf_model.pkl` so it can be reused without retraining the model every time the application runs.

### 4. Model Evaluation

The model's performance can be evaluated using classification metrics such as:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churn cases that are actually churn cases.
- **Recall:** Proportion of actual churn cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes, when valid probability and label data are available.

The project's evaluation results are stored in `model_metrics.json`. Actual metric values should be reported only after checking the saved results.

## 🌐 Streamlit Web Application

The project includes a Streamlit application designed to make the trained model accessible through a user-friendly interface.

Planned application functionality includes:

- **Customer-level prediction:** Enter customer details and obtain a model prediction.
- **Churn probability:** Display the model-estimated probability when supported by the trained pipeline.
- **Model performance:** View the evaluation metrics saved in the project.
- **Business interpretation:** Explain how churn predictions may inform customer retention decisions.

The application should be tested against the original training pipeline before being considered production-ready.

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py                # Streamlit application
├── rf_model.pkl          # Saved trained ML pipeline
├── model_metrics.json    # Model evaluation metrics
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Run the Project Locally

### Prerequisites

Install Python and Git on your computer.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Navigate to the project folder

```bash
cd customer-churn-prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit application

```bash
streamlit run app.py
```

The terminal will display a local URL, usually `http://localhost:8501`, where you can open the application in your browser.

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project files to GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app and select this repository.
4. Select `app.py` as the application entry point.
5. Deploy the application.
6. Test the deployed app with representative customer inputs.

**Deployment note:** The saved model should be loaded using compatible package versions. Confirm that the categorical values and target-class mapping match the original training notebook.

## 💼 Business Applications

Customer churn predictions can support several business activities:

- **Retention campaigns:** Prioritize customers who may need additional support or engagement.
- **Customer segmentation:** Examine patterns across contract types, tenure, and service subscriptions.
- **Revenue protection:** Identify customer groups that may contribute to churn-related revenue loss.
- **Customer experience improvement:** Investigate recurring issues associated with customers predicted to churn.
- **Business reporting:** Combine churn predictions with customer and revenue metrics for further analysis.

A prediction alone does not establish why a customer will churn or prove that a retention campaign will work. Business actions should be evaluated using suitable evidence.

## 🧪 A/B Testing and Retention Strategy

A/B testing can be used to evaluate whether a retention intervention improves customer outcomes.

A possible experiment would compare:

- **Control group:** Customers receive the standard customer experience.
- **Treatment group:** Customers receive a proposed retention intervention.

The experiment can compare churn rates between the two groups and assess whether the difference is statistically meaningful.

If a demonstration uses simulated treatment outcomes, those results must be identified as simulated. They are not evidence of real-world campaign effectiveness. A reliable business conclusion requires actual experiment data and an appropriate experimental design.

## 🔮 Future Improvements

- Add batch predictions for uploaded customer CSV files.
- Improve model evaluation and compare multiple classification algorithms.
- Add model explainability using feature importance or SHAP, where appropriate.
- Build a customer-segment and churn-trend dashboard.
- Evaluate a retention campaign using real A/B test data.
- Monitor model performance and data drift over time.
- Add input validation and automated tests for the deployed application.

## ⚠️ Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- Model-estimated probabilities are not guarantees of future customer behavior.
- Performance on historical data may not generalize to new customer populations.
- Predictions do not establish causality or identify the true reason a customer leaves.
- Retention interventions should be tested before assuming they reduce churn.

## 👨‍💻 Author

**Deepanshu Sonkar**

- GitHub:https://github.com/deepanshunitw-cmyk



