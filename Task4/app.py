import streamlit as st
import joblib
import pandas as pd

loan_model = joblib.load('Task4/random_forest_loan_model.pkl')

st.title("Loan Approval System")
st.write("Enter applicant details for Loan Approval")

gender = st.selectbox("Gender", [0,1], format_func=lambda x: "Male" if x==1 else "female")
married = st.selectbox("Married", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
dependents = st.number_input("Dependents", min_value=0, step=1)
education = st.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x == 1 else "Not Graduate")
self_employed = st.selectbox("Self Employed", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=1)
loan_amount_term = st.number_input("Loan Amount Term", min_value=1)
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])

total_income = applicant_income + coapplicant_income
income_to_loan_ratio = total_income / loan_amount if loan_amount != 0 else 0

if st.button("Predict Loan Approval"):
    input_df = pd.DataFrame([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area,
        total_income, income_to_loan_ratio
    ]], columns=[
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        'Credit_History', 'Property_Area', 'total_income', 'income_to_loan_ratio'
    ])

    prediction = loan_model.predict(input_df)[0]
    if prediction == 1:
        result = "Approved ✅"
    else:
        result = "Not Approved ❌"
    st.subheader(f"Loan Status: {result}")