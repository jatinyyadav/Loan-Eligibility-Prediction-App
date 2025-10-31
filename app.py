import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# ===============================
# Load Model and Columns
# ===============================
model = joblib.load("loan_approval_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ===============================
# Page Setup
# ===============================
st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")

# Modern clean UI styling
page_style = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc;
}

/* Title */
h1, h2, h3, h4, h5, h6 {
    color: #0d47a1 !important;
}

/* Labels */
label, .stTextInput, .stSelectbox label {
    color: #212121 !important;
    font-weight: 500;
}

/* Buttons */
div.stButton > button {
    background-color: #1565c0;
    color: white;
    border: none;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-weight: 600;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #0d47a1;
    color: white;
}

/* Cards */
.block-container {
    padding-top: 2rem;
}
.stTextInput, .stSelectbox, .stNumberInput {
    background: white !important;
    border-radius: 10px !important;
    padding: 0.4rem !important;
    box-shadow: 0px 1px 2px rgba(0,0,0,0.1) !important;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;'>🏦 Loan Eligibility Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("<h5 style='text-align:center; color:#555;'>Enter applicant details to check loan eligibility</h5>", unsafe_allow_html=True)

# ===============================
# Layout Columns
# ===============================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 👤 Personal Information")
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    st.markdown("### 💰 Financial Information")
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)

with col2:
    st.markdown("### 🏠 Loan Details")
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
    Loan_Amount_Term = st.selectbox("Loan Term (in months)", [360, 120, 180, 240, 300, 480])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ===============================
# Prepare Input
# ===============================
input_data = pd.DataFrame({
    "Gender": [Gender],
    "Married": [Married],
    "Dependents": [Dependents],
    "Education": [Education],
    "Self_Employed": [Self_Employed],
    "ApplicantIncome": [ApplicantIncome],
    "CoapplicantIncome": [CoapplicantIncome],
    "LoanAmount": [LoanAmount],
    "Loan_Amount_Term": [Loan_Amount_Term],
    "Credit_History": [Credit_History],
    "Property_Area": [Property_Area]
})

# Align with training columns
input_encoded = pd.get_dummies(input_data)
missing_cols = set(model_columns) - set(input_encoded.columns)
for c in missing_cols:
    input_encoded[c] = 0
input_encoded = input_encoded[model_columns]

# ===============================
# Prediction
# ===============================
st.markdown("---")
if st.button("🔍 Predict Loan Eligibility"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1] * 100

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.success(f"✅ Loan Approved with {probability:.2f}% confidence!")
    else:
        st.error(f"❌ Loan Rejected with {100 - probability:.2f}% confidence.")

    st.progress(int(probability))
    st.markdown(f"**Approval Probability:** {probability:.2f}%")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit & XGBoost</p>",
    unsafe_allow_html=True
)

