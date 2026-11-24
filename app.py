"""
Customer Churn Predictor - live Streamlit app.

A telecom retention team can type in a customer's details and instantly see
their churn risk (High / Medium / Low), the reasons behind the score, and
concrete actions to keep that customer.

The model is the winner from Churn_Project.ipynb:
    Logistic Regression, class_weight="balanced", C=1, random_state=42
    inside a ColumnTransformer pipeline (scaled numbers + one-hot categories).

We refit on the full dataset when the app starts. The notebook uses an 80/20
split for an honest evaluation; the app uses every available row because
there is no test set to protect any more.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration - these values come straight from the Colab notebook results
# ---------------------------------------------------------------------------
C_VALUE = 1                 # best parameter found by GridSearchCV
RANDOM_STATE = 42
HIGH_RISK = 0.65            # churn score >= 65% -> High
MEDIUM_RISK = 0.35          # 35%-65% -> Medium, below 35% -> Low
DATA_PATH = Path(__file__).parent / "telco_churn.csv"

st.set_page_config(
    page_title="Churn Radar | Telco Retention",
    page_icon="📉",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the churn model...")
def load_model():
    """Clean the CSV and fit the exact pipeline that won in Colab."""
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}. Make sure telco_churn.csv is next to app.py.")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    # Same cleaning as the notebook: TotalCharges arrived as text with 11 blanks.
    # Those customers had tenure 0, so a total of 0 is the correct value.
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].str.strip(), errors="coerce"
    ).fillna(0)

    X = data.drop(columns=["customerID", "Churn"])
    y = (data["Churn"] == "Yes").astype(int)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(
                C=C_VALUE,
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]
    )
    model.fit(X, y)
    return model, data, X.columns


model, data, feature_columns = load_model()


# ---------------------------------------------------------------------------
# Sidebar inputs - use the exact category labels from the dataset so the
# one-hot encoder receives the same values it saw during training.
# ---------------------------------------------------------------------------
def yes_no(label, default="No"):
    return st.selectbox(label, ["No", "Yes"], index=0 if default == "No" else 1)


with st.sidebar:
    st.title("🧾 Customer details")
    st.caption("Change any field and the risk score updates instantly.")

    with st.expander("Account", expanded=True):
        tenure = st.slider("Tenure (months)", 0, 72, 2)
        monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.05)
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            value=float(round(tenure * monthly_charges, 2)),
            step=10.0,
            help="Defaults to tenure × monthly charges; adjust if needed.",
        )
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = yes_no("Paperless billing", "Yes")
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with st.expander("Customer profile"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = yes_no("Senior citizen")
        partner = yes_no("Has partner")
        dependents = yes_no("Has dependents")

    with st.expander("Services"):
        phone = yes_no("Phone service", "Yes")
        multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        no_internet = ["Yes", "No", "No internet service"]
        online_security = st.selectbox("Online security", no_internet)
        online_backup = st.selectbox("Online backup", no_internet)
        device_protection = st.selectbox("Device protection", no_internet)
        tech_support = st.selectbox("Tech support", no_internet)
        streaming_tv = st.selectbox("Streaming TV", no_internet)
        streaming_movies = st.selectbox("Streaming movies", no_internet)

    st.divider()
    st.caption("Model: Logistic Regression (balanced), C=1 · Recall 78%")


# Assemble one row with exactly the columns the model was trained on.
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}])[feature_columns]

# The score is a *relative* risk score: class_weight="balanced" shifts the
# probabilities upward, so read it as "how risky compared with an average
# customer", not as a calibrated chance of leaving.
score = float(model.predict_proba(customer)[0, 1])

if score >= HIGH_RISK:
    level, color, icon = "HIGH", "#d63031", "🚨"
elif score >= MEDIUM_RISK:
    level, color, icon = "MEDIUM", "#e17055", "⚠️"
else:
    level, color, icon = "LOW", "#00b894", "✅"


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📉 Churn Radar — Telco Customer Retention")
st.caption(
    "Find out who is likely to leave, why, and what to do about it. "
    "Built from 7,043 real customer records."
)

col1, col2, col3 = st.columns([1.3, 1, 1])
with col1:
    st.markdown(f"### {icon} {level} RISK")
    st.progress(score)
    st.caption(f"Churn score: **{score:.0%}**")
with col2:
    st.metric("Contract", contract)
    st.caption(f"Tenure: {tenure} months")
with col3:
    st.metric("Monthly bill", f"${monthly_charges:,.2f}")
    st.caption(f"Payment: {payment}")

st.caption(
    "ℹ️ The score is relative risk, not a calibrated probability — the model "
    "was deliberately balanced to catch more churners."
)

tab_actions, tab_why, tab_whatif, tab_context = st.tabs(
    ["🎯 Action plan", "🔍 Why this score?", "🧪 What-if simulator", "📊 Business context"]
)


# --- Tab 1: personalised recommendations -----------------------------------
def build_recommendations(row, value):
    """Turn the customer's actual inputs plus the EDA findings into actions."""
    actions = []
    if row["Contract"].iloc[0] == "Month-to-month":
        actions.append((
            "Move this customer to a 1- or 2-year contract",
            "43% of month-to-month customers churn versus 3% on two-year deals. "
            "Offer a small discount or a free upgrade in exchange for signing.",
        ))
    if row["tenure"].iloc[0] <= 12:
        actions.append((
            "Start a first-year retention check-in",
            "53% of customers leave in their first 6 months. Call at months 1, 3 and 6 "
            "to fix problems before they become reasons to leave.",
        ))
    if row["PaymentMethod"].iloc[0] == "Electronic check":
        actions.append((
            "Switch them off electronic check",
            "Electronic-check users churn at 45% versus 17% on bank autopay. "
            "A one-off bill credit usually covers the cost of the switch.",
        ))
    if row["InternetService"].iloc[0] == "Fiber optic":
        actions.append((
            "Bundle Tech Support or Online Security for free",
            "Fibre customers churn at 42% versus 19% on DSL — support concerns "
            "are a common cause. A 3-month trial is cheap insurance.",
        ))
    if value >= MEDIUM_RISK:
        actions.append((
            "Flag the account for the retention team this week",
            f"A score of {value:.0%} is in the top risk band. Proactive outreach "
            "costs far less than replacing the customer.",
        ))
    if not actions:
        actions.append((
            "No urgent action needed",
            "This profile resembles customers who stay. Keep monitoring at renewal "
            "and continue normal onboarding.",
        ))
    return actions


with tab_actions:
    st.subheader("Recommended retention actions")
    st.caption("Based on the customer's details and the churn patterns found in the EDA.")
    for i, (title, why) in enumerate(build_recommendations(customer, score), start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {title}**")
            st.caption(why)

with tab_why:
    st.subheader("What is driving this score?")
    st.caption(
        "Logistic regression gives each encoded feature a weight. Positive weights "
        "push towards churn; negative weights push towards staying. Numbers were "
        "standardised and categories one-hot encoded, so the weights are comparable."
    )
    coefficients = model.named_steps["model"].coef_[0]
    feature_names = model.named_steps["prep"].get_feature_names_out()
    top = np.argsort(np.abs(coefficients))[-12:]
    top = top[np.argsort(coefficients[top])]

    names = [feature_names[i].replace("num__", "").replace("cat__", "") for i in top]
    values = coefficients[top]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color=["#00b894" if v < 0 else "#d63031" for v in values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effect on churn score (log-odds)")
    ax.set_title("Strongest model drivers")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Note: tenure, total charges and monthly charges are correlated, so their "
        "individual weights partly offset each other. Read the direction of the "
        "strongest features, not each number in isolation."
    )

with tab_whatif:
    st.subheader("What if we changed one thing?")
    st.caption("See how the score moves under realistic retention offers.")
    scenarios = {
        "Current profile": {},
        "Two-year contract": {"Contract": "Two year"},
        "Automatic bank payment": {"PaymentMethod": "Bank transfer (automatic)"},
        "Both changes": {
            "Contract": "Two year",
            "PaymentMethod": "Bank transfer (automatic)",
        },
    }
    sim_rows = []
    for name, changes in scenarios.items():
        trial = customer.copy()
        for key, value in changes.items():
            trial[key] = value
        sim_rows.append({"Scenario": name, "Churn score": model.predict_proba(trial)[0, 1]})
    sim = pd.DataFrame(sim_rows).set_index("Scenario")
    sim["Change"] = sim["Churn score"] - score
    st.dataframe(sim.style.format({"Churn score": "{:.1%}", "Change": "{:+.1%}"}))
    st.bar_chart(sim["Churn score"])
    st.caption("A negative change means the intervention lowers churn risk.")

with tab_context:
    st.subheader("The bigger picture")
    churn_flag = (data["Churn"] == "Yes").astype(int)
    by_contract = data.assign(_churn=churn_flag).groupby("Contract")["_churn"].mean()
    by_payment = data.assign(_churn=churn_flag).groupby("PaymentMethod")["_churn"].mean()

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.caption("**Churn rate by contract**")
        st.bar_chart(by_contract)
    with chart_col2:
        st.caption("**Churn rate by payment method**")
        st.bar_chart(by_payment)

    st.markdown(
        """
        **Key findings from the notebook**

        - Month-to-month customers churn at **43%**, versus **3%** on two-year contracts.
        - Churn is concentrated early: **53%** in the first 6 months.
        - Electronic-check users churn at **45%**, versus **17%** on automatic bank payment.
        - Fibre customers churn at **42%**, versus **19%** on DSL.

        **Model performance (held-out test set)**

        - Recall **0.78** — we catch 78% of customers who actually leave.
        - Accuracy **0.74**, precision **0.50**, ROC-AUC **0.84**.
        - Confusion matrix `[[747, 288], [81, 293]]`: 81 churners missed,
          288 extra retention offers — a trade-off we chose on purpose.
        """
    )

with st.expander("How does this app work?"):
    st.markdown(
        """
        1. `telco_churn.csv` is cleaned exactly as in the notebook (text `TotalCharges`
           converted to numbers, 11 blanks filled with 0, `customerID` dropped).
        2. A `ColumnTransformer` scales numeric columns and one-hot encodes categories.
        3. A balanced Logistic Regression (`C=1`, `random_state=42`) is refitted on all
           7,043 rows — the winning setup from `GridSearchCV` in the notebook.
        4. Your inputs are passed through the same pipeline and scored.
        5. `class_weight="balanced"` makes the model pay more attention to churners,
           which raises recall: missing a churner costs more than an extra offer.
        """
    )

st.divider()
st.caption(
    "Built with Python, pandas, scikit-learn and Streamlit · "
    "Data: Telco Customer Churn (Kaggle) · Model and analysis: Churn_Project.ipynb"
)
