"""
Customer Churn Predictor - live Streamlit app.

This is a SCORING TOOL for new, unseen customer data:
  * Score one customer interactively and see why they are at risk.
  * Upload a CSV of customers and download their risk scores in bulk.

The model is the winner from Churn_Project.ipynb:
    Logistic Regression, class_weight="balanced", C=1, random_state=42
    inside a ColumnTransformer pipeline (scaled numbers + one-hot categories).

The notebook uses an 80/20 split for honest evaluation. This app refits the
same pipeline on all available rows so it is ready to score new customers.
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
# Configuration - taken directly from the Colab notebook results
# ---------------------------------------------------------------------------
C_VALUE = 1                 # best parameter found by GridSearchCV
RANDOM_STATE = 42
HIGH_RISK = 0.65            # score >= 65% -> High
MEDIUM_RISK = 0.35          # 35%-65% -> Medium, below 35% -> Low
DATA_PATH = Path(__file__).parent / "telco_churn.csv"

st.set_page_config(
    page_title="Churn Radar | Telco Retention",
    page_icon="📉",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data + model (cached: trained once per app session, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the churn model...")
def load_model():
    """Clean the CSV and fit the exact pipeline that won in Colab."""
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}. Keep telco_churn.csv next to app.py.")
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
    return model, data, X.columns.tolist(), numeric


model, data, feature_columns, numeric_columns = load_model()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def risk_label(score):
    if score >= HIGH_RISK:
        return "High"
    if score >= MEDIUM_RISK:
        return "Medium"
    return "Low"


def top_action(row):
    """One-line action for a batch of customers, based on the EDA findings."""
    if row["Contract"] == "Month-to-month":
        return "Offer a 1- or 2-year contract"
    if row["PaymentMethod"] == "Electronic check":
        return "Switch to automatic payment"
    if row["tenure"] <= 12:
        return "First-year retention check-in"
    if row["InternetService"] == "Fiber optic":
        return "Bundle free Tech Support"
    return "Monitor at renewal"


def default_or_none(key, fallback):
    """Avoid Streamlit's 'default + session state' warning on preset changes."""
    return None if key in st.session_state else fallback


def apply_high_preset():
    """Set a deliberately risky example profile before widgets are rebuilt."""
    st.session_state.update({
        "tenure": 2, "monthly": 85.0, "contract": "Month-to-month",
        "payment": "Electronic check", "internet": "Fiber optic",
        "online_security": "No", "tech_support": "No",
        "gender": "Male", "senior": "Yes", "partner": "No",
        "dependents": "No", "phone": "Yes", "multiple_lines": "Yes",
        "paperless": "Yes", "online_backup": "No",
        "device_protection": "No", "streaming_tv": "Yes",
        "streaming_movies": "Yes",
    })


def apply_low_preset():
    """Set a low-risk example profile before widgets are rebuilt."""
    st.session_state.update({
        "tenure": 60, "monthly": 45.0, "contract": "Two year",
        "payment": "Bank transfer (automatic)", "internet": "DSL",
        "online_security": "Yes", "tech_support": "Yes",
        "gender": "Female", "senior": "No", "partner": "Yes",
        "dependents": "Yes", "phone": "Yes", "multiple_lines": "No",
        "paperless": "No", "online_backup": "Yes",
        "device_protection": "Yes", "streaming_tv": "No",
        "streaming_movies": "No",
    })


def draw_gauge(score):
    """A small colour-banded gauge: green low, amber medium, red high."""
    fig, ax = plt.subplots(figsize=(9, 1.1))
    ax.barh(0, MEDIUM_RISK, height=0.45, color="#00b894")
    ax.barh(0, HIGH_RISK - MEDIUM_RISK, left=MEDIUM_RISK, height=0.45, color="#f39c12")
    ax.barh(0, 1 - HIGH_RISK, left=HIGH_RISK, height=0.45, color="#d63031")
    ax.axvline(score, color="black", linewidth=3)
    ax.text(score, 0.42, f" {score:.0%}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 0.6)
    ax.set_xticks([0, MEDIUM_RISK, HIGH_RISK, 1])
    ax.set_xticklabels(["0%", "35%", "65%", "100%"])
    ax.set_yticks([])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page header + mode selector
# ---------------------------------------------------------------------------
st.title("📉 Churn Radar — Telco Customer Retention")
st.caption(
    "Score new customer data and get clear, actionable retention guidance. "
    "Built from 7,043 real customer records."
)

mode = st.radio(
    "What would you like to do?",
    ["🔍 Score one customer", "📂 Upload a CSV of customers"],
    horizontal=True,
    label_visibility="collapsed",
)
st.caption(
    "Scores are relative risk, not calibrated probabilities — the model was "
    "deliberately balanced to catch more churners."
)


# ===========================================================================
# MODE 1 - score a single customer from live inputs
# ===========================================================================
if mode == "🔍 Score one customer":

    def yes_no(label, default="No", key=None):
        index = None if key in st.session_state else (0 if default == "No" else 1)
        return st.selectbox(label, ["No", "Yes"], index=index, key=key)

    with st.sidebar:
        st.title("🧾 Customer details")
        st.caption("Change any field and the score updates instantly.")

        with st.expander("Account", expanded=True):
            tenure = st.slider(
                "Tenure (months)", 0, 72, default_or_none("tenure", 2), key="tenure")
            monthly_charges = st.slider(
                "Monthly charges ($)", 18.0, 120.0,
                default_or_none("monthly", 70.0), step=0.05, key="monthly",
            )
            total_charges = round(tenure * monthly_charges, 2)
            st.caption(f"Estimated total charges: **${total_charges:,.2f}** "
                       "(tenure × monthly charges)")
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"],
                index=default_or_none("contract", 0), key="contract",
            )
            paperless = yes_no("Paperless billing", "Yes", key="paperless")
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
                index=default_or_none("payment", 0), key="payment",
            )

        with st.expander("Customer profile"):
            gender = st.selectbox(
                "Gender", ["Female", "Male"], index=default_or_none("gender", 0), key="gender")
            senior = yes_no("Senior citizen", key="senior")
            partner = yes_no("Has partner", key="partner")
            dependents = yes_no("Has dependents", key="dependents")

        with st.expander("Services"):
            phone = yes_no("Phone service", "Yes", key="phone")
            multiple_lines = st.selectbox(
                "Multiple lines", ["Yes", "No", "No phone service"],
                index=default_or_none("multiple_lines", 0), key="multiple_lines")
            internet = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"],
                index=default_or_none("internet", 0), key="internet")
            no_internet = ["Yes", "No", "No internet service"]
            online_security = st.selectbox(
                "Online security", no_internet,
                index=default_or_none("online_security", 1), key="online_security")
            online_backup = st.selectbox(
                "Online backup", no_internet,
                index=default_or_none("online_backup", 0), key="online_backup")
            device_protection = st.selectbox(
                "Device protection", no_internet,
                index=default_or_none("device_protection", 0), key="device_protection")
            tech_support = st.selectbox(
                "Tech support", no_internet,
                index=default_or_none("tech_support", 1), key="tech_support")
            streaming_tv = st.selectbox(
                "Streaming TV", no_internet,
                index=default_or_none("streaming_tv", 0), key="streaming_tv")
            streaming_movies = st.selectbox(
                "Streaming movies", no_internet,
                index=default_or_none("streaming_movies", 0), key="streaming_movies")

        st.divider()
        st.caption("Try an example:")
        example_col1, example_col2 = st.columns(2)
        example_col1.button("⚠️ High risk", key="btn_high", on_click=apply_high_preset)
        example_col2.button("✅ Low risk", key="btn_low", on_click=apply_low_preset)

    # One row, in exactly the training column order.
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

    score = float(model.predict_proba(customer)[0, 1])
    level = risk_label(score)

    risk_col, fact_col1, fact_col2 = st.columns([2.2, 1, 1])
    with risk_col:
        st.markdown(f"### {'🚨' if level == 'High' else '⚠️' if level == 'Medium' else '✅'} {level.upper()} RISK")
        st.pyplot(draw_gauge(score))
    with fact_col1:
        st.metric("Churn score", f"{score:.0%}",
                  delta=f"{score - 0.265:+.0%} vs average customer")
        st.caption(f"Contract: **{contract}**")
    with fact_col2:
        st.metric("Monthly bill", f"${monthly_charges:,.2f}")
        st.caption(f"Tenure: **{tenure} months** · {payment}")

    tab_actions, tab_why, tab_whatif, tab_context = st.tabs(
        ["🎯 Action plan", "🔍 Why this score?", "🧪 What-if simulator", "📊 Business context"]
    )

    with tab_actions:
        st.subheader("Recommended retention actions")
        st.caption("Driven by this customer's inputs and the patterns found in the EDA.")

        actions = []
        if contract == "Month-to-month":
            actions.append((
                "Move this customer to a 1- or 2-year contract",
                "43% of month-to-month customers churn versus 3% on two-year deals. "
                "Offer a discount or free upgrade in exchange for signing.",
            ))
        if tenure <= 12:
            actions.append((
                "Start a first-year retention check-in",
                "53% of customers leave in their first 6 months. Call at months 1, 3 and 6.",
            ))
        if payment == "Electronic check":
            actions.append((
                "Switch them off electronic check",
                "Electronic-check users churn at 45% versus 17% on bank autopay. "
                "A one-off bill credit is cheaper than losing the customer.",
            ))
        if internet == "Fiber optic":
            actions.append((
                "Bundle Tech Support or Online Security for free",
                "Fibre customers churn at 42% versus 19% on DSL. A 3-month trial is cheap insurance.",
            ))
        if score >= MEDIUM_RISK:
            actions.append((
                "Flag the account for the retention team this week",
                f"A score of {score:.0%} sits in the top risk bands — reach out proactively.",
            ))
        if not actions:
            actions.append((
                "No urgent action needed",
                "This profile resembles customers who stay. Keep monitoring at renewal.",
            ))

        for i, (title, why) in enumerate(actions, start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {title}**")
                st.caption(why)

    with tab_why:
        st.subheader("What is driving this score?")
        st.caption(
            "Logistic regression gives each encoded feature a weight. Positive weights "
            "push towards churn; negative weights push towards staying. Numeric features "
            "were standardised and categories one-hot encoded, so weights are comparable."
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
            "Tenure, total charges and monthly charges are correlated, so their weights "
            "partly offset each other. Read the direction of the strongest features."
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

            - Month-to-month: **43%** churn vs **3%** on two-year contracts.
            - **53%** of churn happens in the first 6 months.
            - Electronic check: **45%** vs **17%** on automatic bank payment.
            - Fibre: **42%** vs **19%** on DSL.

            **Held-out test performance:** recall **0.78**, accuracy **0.74**,
            precision **0.50**, ROC-AUC **0.84**.
            """
        )


# ===========================================================================
# MODE 2 - score a batch of new customers from an uploaded CSV
# ===========================================================================
else:
    st.subheader("Score a CSV of customers")
    st.caption(
        "Upload any CSV containing the customer fields listed below. Each row is "
        "scored independently, and you can download the results."
    )

    with st.expander("Required CSV columns"):
        st.code(", ".join(feature_columns), language="text")
        st.caption(
            "Extra columns (such as customerID) are fine — they are ignored for scoring. "
            "Unknown categories are handled safely by the model."
        )

    template = data.drop(columns=["Churn"]).head(5)
    st.download_button(
        "⬇️ Download an example CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="churn_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}")
            st.stop()

        missing = [col for col in feature_columns if col not in batch.columns]
        if missing:
            st.error("Your CSV is missing these columns: " + ", ".join(missing))
            st.stop()

        # Same cleaning rules as the notebook / single-customer mode.
        batch["TotalCharges"] = pd.to_numeric(
            batch["TotalCharges"].astype(str).str.strip(), errors="coerce"
        ).fillna(0)
        for col in numeric_columns:
            batch[col] = pd.to_numeric(batch[col], errors="coerce").fillna(0)

        probabilities = model.predict_proba(batch[feature_columns])[:, 1]
        results = batch.copy()
        results.insert(0, "ChurnScore", probabilities)
        results.insert(1, "RiskLevel", [risk_label(p) for p in probabilities])
        results.insert(2, "RecommendedAction", batch.apply(top_action, axis=1))

        counts = results["RiskLevel"].value_counts()
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Customers scored", len(results))
        metric_col2.metric("🚨 High risk", int(counts.get("High", 0)))
        metric_col3.metric("⚠️ Medium risk", int(counts.get("Medium", 0)))
        metric_col4.metric("✅ Low risk", int(counts.get("Low", 0)))

        filter_level = st.selectbox("Show only", ["All", "High", "Medium", "Low"])
        shown = results if filter_level == "All" else results[results["RiskLevel"] == filter_level]
        st.dataframe(
            shown.style.format({"ChurnScore": "{:.1%}"}),
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download predictions",
            results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )
        st.caption("Tip: filter to High risk first — those are the customers to call this week.")
    else:
        st.info("Upload a CSV above to get started, or download the example template to see the format.")


# ---------------------------------------------------------------------------
# Shared explainer + footer
# ---------------------------------------------------------------------------
with st.expander("How does this app work?"):
    st.markdown(
        """
        1. `telco_churn.csv` is cleaned exactly as in the notebook: text `TotalCharges`
           becomes numeric, 11 blanks (tenure 0) become 0, `customerID` is dropped.
        2. A `ColumnTransformer` scales numeric columns and one-hot encodes categories
           with `handle_unknown="ignore"`, so new categories never break scoring.
        3. Balanced Logistic Regression (`C=1`, `random_state=42`) — the winner from
           `GridSearchCV` in the notebook — is refitted on all 7,043 rows.
        4. New data goes through the *same* pipeline, so training and scoring match.
        5. `class_weight="balanced"` makes the model pay more attention to churners:
           recall 0.78 means it catches 78% of customers who actually leave.
        """
    )

st.divider()
st.caption(
    "Built with Python, pandas, scikit-learn and Streamlit · "
    "Data: Telco Customer Churn (Kaggle) · Model and analysis: Churn_Project.ipynb"
)
