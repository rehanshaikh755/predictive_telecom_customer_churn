"""ConnectTel Churn Predictor (Streamlit).

The model is rebuilt from the Telco CSV when the app starts, using the tuned settings from the project
notebook (same 80/20 stratified split and seed). This avoids scikit-learn pickle-version problems.
Needs WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder as this file.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="ConnectTel Churn Predictor", page_icon="📉")

CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
THRESHOLD = 0.26  # profit-optimal threshold found in the notebook (placeholder cost assumptions)
SEED = 42


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


def find_csv():
    for p in (CSV_NAME, os.path.join("data", CSV_NAME)):
        if os.path.exists(p):
            return p
    return None


def make_classifier(pos_weight):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, min_child_weight=1,
                             subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                             eval_metric="auc", n_jobs=1, random_state=SEED), "XGBoost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, max_iter=200,
                                              class_weight="balanced", random_state=SEED), "Gradient boosting (sklearn fallback)"


@st.cache_resource(show_spinner="Preparing the model (first load only)...")
def load_model():
    path = find_csv()
    if path is None:
        return None
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0.0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df = engineer_features(df.drop(columns=["customerID"]))
    X, y = df.drop(columns=["Churn"]), df["Churn"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat),
    ], sparse_threshold=0)
    clf, name = make_classifier((y_train == 0).sum() / (y_train == 1).sum())
    pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_train, y_train)
    return {"pipeline": pipe, "features": X.columns.tolist(), "name": name}


# ---- UI ----
st.title("ConnectTel Churn Predictor")
st.caption("Enter a customer profile to estimate churn risk.")

bundle = load_model()
if bundle is None:
    st.error(f"{CSV_NAME} not found. Upload it to the same folder as streamlit_app.py in the GitHub repo.")
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
    st.metric("Churn probability", f"{p:.1%}")
    st.progress(min(p, 1.0))
    if p >= THRESHOLD:
        st.error(f"HIGH RISK (above threshold {THRESHOLD:.2f}): recommend a retention offer.")
    else:
        st.success(f"Lower risk (below threshold {THRESHOLD:.2f}).")
    st.caption(f"Model: {bundle['name']}")
