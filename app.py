"""
Customer Churn Predictor - clean scoring interface for new customer data.

The model is the winner from Churn_Project.ipynb (balanced Logistic
Regression, C=1) and is refitted from telco_churn.csv on app startup.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

C_VALUE = 1
RANDOM_STATE = 42
HIGH_RISK = 0.65
MEDIUM_RISK = 0.35
DATA_PATH = Path(__file__).parent / "telco_churn.csv"

st.set_page_config(page_title="Churn Risk Checker", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].str.strip(), errors="coerce"
    ).fillna(0)

    X = data.drop(columns=["customerID", "Churn"])
    y = (data["Churn"] == "Yes").astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         X.select_dtypes(exclude="number").columns.tolist()),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            C=C_VALUE, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE)),
    ])
    model.fit(X, y)
    return model, data, X.columns.tolist(), numeric


model, data, feature_columns, numeric_columns = load_model()


def risk_label(score):
    if score >= HIGH_RISK:
        return "High"
    if score >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def action_for(row):
    if row["Contract"] == "Month-to-month":
        return "Offer a 1- or 2-year contract"
    if row["PaymentMethod"] == "Electronic check":
        return "Switch to automatic payment"
    if row["tenure"] <= 12:
        return "First-year retention check-in"
    if row["InternetService"] == "Fiber optic":
        return "Bundle free Tech Support"
    return "Monitor at renewal"


def prepare_scoring(frame):
    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(
        frame["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0)
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    for col in feature_columns:
        if col not in numeric_columns:
            frame[col] = frame[col].astype(str)
    return frame


st.title("Churn Risk Checker")
st.caption("Enter customer details to check churn risk and get retention actions.")

view = st.radio(
    "Input mode",
    ["Single customer", "Fill a table", "Upload CSV"],
    horizontal=True,
    label_visibility="collapsed",
)


# ---------------------------------------------------------------- single form
if view == "Single customer":
    with st.form("customer_form"):
        st.markdown("#### Customer details")
        a, b, c, d = st.columns(4)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with b:
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
        with c:
            partner = st.selectbox("Partner", ["No", "Yes"])
        with d:
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        e, f, g, h = st.columns(4)
        with e:
            tenure = st.number_input("Tenure (months)", 0, 72, 2)
        with f:
            monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, 70.0)
        with g:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with h:
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        i, j, k, l = st.columns(4)
        with i:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        with j:
            security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        with k:
            support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        with l:
            paperless = st.selectbox("Paperless billing", ["No", "Yes"])

        m, n, o, p = st.columns(4)
        with m:
            phone = st.selectbox("Phone service", ["No", "Yes"])
        with n:
            lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        with o:
            backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        with p:
            protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])

        r, s, t, u = st.columns(4)
        with r:
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with s:
            movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Check churn risk", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": round(tenure * monthly, 2),
        }])[feature_columns]

        score = float(model.predict_proba(row)[0, 1])
        level = risk_label(score)
        icon = {"High": "🚨", "Medium": "⚠️", "Low": "✅"}[level]

        st.markdown("---")
        st.markdown(f"### {icon} {level} risk · {score:.0%}")
        st.progress(score)
        st.markdown("#### Recommended actions")
        for item in [
            "Offer a 1- or 2-year contract" if contract == "Month-to-month" else None,
            "First-year retention check-in" if tenure <= 12 else None,
            "Switch to automatic payment" if payment == "Electronic check" else None,
            "Bundle free Tech Support" if internet == "Fiber optic" else None,
            "Flag the account for the retention team" if score >= MEDIUM_RISK else None,
        ]:
            if item:
                st.markdown(f"- {item}")


# ------------------------------------------------------------- editable table
elif view == "Fill a table":
    defaults = {}
    for col in feature_columns:
        if col in numeric_columns:
            defaults[col] = 0
        else:
            defaults[col] = sorted(data[col].astype(str).unique())[0]

    template = pd.DataFrame([dict(defaults) for _ in range(3)])
    config = {}
    for col in feature_columns:
        if col in numeric_columns:
            config[col] = st.column_config.NumberColumn(
                format="%d" if col in ("SeniorCitizen", "tenure") else "%.2f")
        else:
            config[col] = st.column_config.SelectboxColumn(
                options=sorted(data[col].astype(str).unique().tolist()))

    st.markdown("#### Enter customers")
    edited = st.data_editor(template, column_config=config, num_rows="dynamic")

    if st.button("Score customers", type="primary"):
        if edited.empty:
            st.warning("Add at least one customer row.")
        else:
            frame = prepare_scoring(edited[feature_columns])
            scores = model.predict_proba(frame)[:, 1]
            out = frame.copy()
            out.insert(0, "Action", frame.apply(action_for, axis=1))
            out.insert(0, "Score", [f"{s:.0%}" for s in scores])
            out.insert(0, "Risk", [risk_label(s) for s in scores])
            st.markdown("#### Results")
            st.dataframe(out[["Risk", "Score", "Action", "Contract", "tenure",
                              "MonthlyCharges", "PaymentMethod"]], hide_index=True)
            st.download_button(
                "Download results",
                out.to_csv(index=False).encode("utf-8"),
                file_name="churn_results.csv", mime="text/csv",
            )


# ------------------------------------------------------------------ CSV upload
else:
    st.markdown("#### Upload customers")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        incoming = pd.read_csv(uploaded)
        missing = [col for col in feature_columns if col not in incoming.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            frame = prepare_scoring(incoming[feature_columns])
            scores = model.predict_proba(frame)[:, 1]
            out = frame.copy()
            out.insert(0, "Action", frame.apply(action_for, axis=1))
            out.insert(0, "Score", [f"{s:.0%}" for s in scores])
            out.insert(0, "Risk", [risk_label(s) for s in scores])
            st.markdown("#### Results")
            st.dataframe(out[["Risk", "Score", "Action", "Contract", "tenure",
                              "MonthlyCharges", "PaymentMethod"]], hide_index=True)
            st.download_button(
                "Download results",
                out.to_csv(index=False).encode("utf-8"),
                file_name="churn_results.csv", mime="text/csv",
            )
