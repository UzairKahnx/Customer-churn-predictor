# Customer Churn Predictor

**Live app:** https://customer-churn-predictor-kzik33kexpbcymqrathyok.streamlit.app/

Predicts which telecom customers are likely to leave, explains why, and suggests concrete actions to keep them.

## The problem

A telecom company loses customers every month. Retaining a customer costs far less than acquiring a new one, so the goal was to find the *why* behind churn and turn it into a tool the retention team can actually use.

## Key findings

- Month-to-month customers churn at **43%**, compared with **3%** on two-year contracts.
- Churn is front-loaded: **53%** in the first 6 months, falling to 10% after 4 years.
- Electronic-check users churn at **45%**, versus 17% on automatic bank payment.
- Fibre customers churn at **42%**, versus 19% on DSL.
- A balanced model catches **78%** of churners (recall 0.783) at 74% accuracy.

## Model

The notebook compares Logistic Regression, Decision Tree and Random Forest, with and without `class_weight="balanced"`. Because a missed churner is a lost customer, **recall** was the decision metric.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression (balanced)** | 0.738 | 0.504 | **0.783** | 0.614 |
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 |
| Random Forest | 0.786 | 0.625 | 0.487 | 0.547 |
| Decision Tree | 0.721 | 0.475 | 0.492 | 0.484 |

`GridSearchCV` (5-fold, scoring = recall) kept `C=1`. On the held-out 20% test set:

- **Recall 0.783** · Precision 0.504 · Accuracy 0.738 · F1 0.614 · ROC-AUC 0.842
- Confusion matrix `[[747, 288], [81, 293]]` — 81 churners missed, 288 extra retention offers. That trade-off is intentional.

Pipeline: `StandardScaler` for numeric features + `OneHotEncoder(handle_unknown="ignore")` for categorical features, feeding `LogisticRegression(C=1, class_weight="balanced", random_state=42)`. The app refits this exact pipeline on all available rows.

## Business recommendations

1. **Move month-to-month customers onto longer contracts** — offer a discount for a 1- or 2-year commitment (43% → 3% churn).
2. **Act in the first year** — proactive check-ins at months 1, 3 and 6; over half of churn happens in the first 6 months.
3. **Nudge electronic-check users to autopay** — a small bill credit is cheaper than losing them (45% → 17% churn).
4. **Bundle support with fibre** — a free Tech Support / Online Security trial addresses the highest-churn internet segment (42% → 19% churn).

## Using the app

The app is a **scoring tool for new, unseen customer data**. It offers three clean input modes:

- **Single customer** — fill in a compact form and submit to get the risk level and recommended actions.
- **Fill a table** — enter customer rows directly in an editable table, then score them all at once.
- **Upload CSV** — score a file of customers and download the results.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: live single-customer scoring + batch CSV scoring |
| `Churn_Project.ipynb` | Full 11-step analysis and model training (Colab) |
| `telco_churn.csv` | Dataset (Kaggle Telco Customer Churn, 7,043 customers) |
| `requirements.txt` | Python dependencies |
| `linkedin_post.md` | Draft post for sharing the project |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Built with

Python · pandas · scikit-learn · matplotlib · Streamlit. Data from the Kaggle *Telco Customer Churn* dataset.
