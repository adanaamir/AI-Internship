import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'random_forest_loan_model.pkl'
loan_model = joblib.load(MODEL_PATH)

st.title("Loan Approval System")
st.write("Enter applicant details to predict loan approval.")

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender",       ["Male", "Female"])
    married         = st.selectbox("Married",      ["Yes", "No"])
    dependents      = st.selectbox("Dependents",   ["0", "1", "2", "3+"])
    education       = st.selectbox("Education",    ["Graduate", "Not Graduate"])
    self_employed   = st.selectbox("Self Employed",["Yes", "No"])
    property_area   = st.selectbox("Property Area",["Urban", "Semiurban", "Rural"])

with col2:
    applicant_income   = st.number_input("Applicant Income",   min_value=0,   step=500,  value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0,   step=500,  value=0)
    loan_amount        = st.number_input("Loan Amount",        min_value=1,   step=10,   value=100)
    loan_amount_term   = st.number_input("Loan Amount Term",   min_value=1,   step=12,   value=360)
    credit_history     = st.selectbox("Credit History",        [1.0, 0.0],
                                      format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)")

total_income        = applicant_income + coapplicant_income
income_to_loan_ratio = total_income / loan_amount if loan_amount != 0 else 0

if st.button("Predict Loan Approval"):
    input_df = pd.DataFrame([[
        gender, dependents, self_employed, property_area, married, education,
        float(loan_amount), float(total_income), float(income_to_loan_ratio),
        float(loan_amount_term), credit_history,
        float(applicant_income), float(coapplicant_income),
    ]], columns=[
        'Gender', 'Dependents', 'Self_Employed', 'Property_Area', 'Married', 'Education',
        'LoanAmount', 'total_income', 'income_to_loan_ratio',
        'Loan_Amount_Term', 'Credit_History',
        'ApplicantIncome', 'CoapplicantIncome',
    ])

    prediction = loan_model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")