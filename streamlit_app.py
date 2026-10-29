"""Streamlit demo: score a customer with the saved ConnectTel churn model.
Run:  streamlit run app.py      (needs connecttel_churn_model.joblib from the notebook)
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ConnectTel Churn Predictor", page_icon="📉")

@st.cache_resource
def load():
    return joblib.load("connecttel_churn_model.joblib")

def engineer_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    d["TotalChargesPerTenure"] = d["TotalCharges"] / d["tenure"].clip(lower=1)
    d["ServiceCount"] = sum((d[c] == "Yes").astype(int) for c in service_cols)
    d["ChargeDelta"] = d["MonthlyCharges"] - d["TotalChargesPerTenure"]
    d["IsMonthToMonth"] = (d["Contract"] == "Month-to-month").astype(int)
    d["HasFamily"] = ((d["Partner"] == "Yes") | (d["Dependents"] == "Yes")).astype(int)
    d["AutoPay"] = d["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    return d

st.title("ConnectTel Churn Predictor")
st.caption("Enter a customer profile to estimate churn risk.")

try:
    bundle = load()
except FileNotFoundError:
    st.error("connecttel_churn_model.joblib not found. Run the notebook (last cell) first.")
    st.stop()

yn = ["Yes", "No"]
c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", yn)
    dependents = st.selectbox("Dependents", yn)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", yn)
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", 15.0, 130.0, 70.0)
with c2:
    phone = st.selectbox("Phone service", yn)
    lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    opts = ["No", "Yes"] if internet != "No" else ["No internet service"]
    sec = st.selectbox("Online security", opts)
    bkp = st.selectbox("Online backup", opts)
    dev = st.selectbox("Device protection", opts)
    tech = st.selectbox("Tech support", opts)
    tv = st.selectbox("Streaming TV", opts)
    mov = st.selectbox("Streaming movies", opts)

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
    "OnlineSecurity": sec, "OnlineBackup": bkp, "DeviceProtection": dev, "TechSupport": tech,
    "StreamingTV": tv, "StreamingMovies": mov, "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
}])

if st.button("Predict churn risk", type="primary"):
    X = engineer_features(row)[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(X)[0, 1])
    thr = bundle["threshold"]
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(min(p, 1.0))
    if p >= thr:
        st.error(f"HIGH RISK (above threshold {thr:.2f}): recommend a retention offer.")
    else:
        st.success(f"Lower risk (below threshold {thr:.2f}).")

