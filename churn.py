import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("customer_churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Customer Churn Prediction")

st.title("📊 Customer Churn Prediction")

# Numerical Inputs
senior = st.selectbox("Senior Citizen", [0, 1])

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

# Binary Inputs
gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone = st.selectbox("Phone Service", ["No", "Yes"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

# Service Inputs
multiple = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Predict Button
if st.button("Predict Churn"):

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical Features
    input_df["SeniorCitizen"] = senior
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly
    input_df["TotalCharges"] = total

    # Binary Features
    if gender == "Male":
        input_df["gender_Male"] = 1

    if partner == "Yes":
        input_df["Partner_Yes"] = 1

    if dependents == "Yes":
        input_df["Dependents_Yes"] = 1

    if phone == "Yes":
        input_df["PhoneService_Yes"] = 1

    if paperless == "Yes":
        input_df["PaperlessBilling_Yes"] = 1

    # Multiple Lines
    if multiple == "No phone service":
        input_df["MultipleLines_No phone service"] = 1
    elif multiple == "Yes":
        input_df["MultipleLines_Yes"] = 1

    # Internet Service
    if internet == "Fiber optic":
        input_df["InternetService_Fiber optic"] = 1
    elif internet == "No":
        input_df["InternetService_No"] = 1

    # Online Security
    if security == "No internet service":
        input_df["OnlineSecurity_No internet service"] = 1
    elif security == "Yes":
        input_df["OnlineSecurity_Yes"] = 1

    # Online Backup
    if backup == "No internet service":
        input_df["OnlineBackup_No internet service"] = 1
    elif backup == "Yes":
        input_df["OnlineBackup_Yes"] = 1

    # Device Protection
    if device == "No internet service":
        input_df["DeviceProtection_No internet service"] = 1
    elif device == "Yes":
        input_df["DeviceProtection_Yes"] = 1

    # Tech Support
    if support == "No internet service":
        input_df["TechSupport_No internet service"] = 1
    elif support == "Yes":
        input_df["TechSupport_Yes"] = 1

    # Streaming TV
    if tv == "No internet service":
        input_df["StreamingTV_No internet service"] = 1
    elif tv == "Yes":
        input_df["StreamingTV_Yes"] = 1

    # Streaming Movies
    if movies == "No internet service":
        input_df["StreamingMovies_No internet service"] = 1
    elif movies == "Yes":
        input_df["StreamingMovies_Yes"] = 1

    # Contract
    if contract == "One year":
        input_df["Contract_One year"] = 1
    elif contract == "Two year":
        input_df["Contract_Two year"] = 1

    # Payment Method
    if payment == "Credit card (automatic)":
        input_df["PaymentMethod_Credit card (automatic)"] = 1

    elif payment == "Electronic check":
        input_df["PaymentMethod_Electronic check"] = 1

    elif payment == "Mailed check":
        input_df["PaymentMethod_Mailed check"] = 1

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer Likely to Churn")
    else:
        st.success("✅ Customer Likely to Stay")

    st.write(
        f"Churn Probability: {probability:.2%}"
    )