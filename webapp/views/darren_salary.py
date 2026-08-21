"""Darren Chor: salary prediction page.

Single and batch prediction with the CatBoost regression model trained on the
global AI and data jobs dataset. The model predicts annual base salary in USD.
"""

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PKL = ROOT / "darren" / "models" / "final_salary_catboost_pipeline.pkl"
RAW_CSV = ROOT / "darren" / "data" / "raw" / "global_ai_jobs.csv"

FEATURES = [
    "country", "job_role", "ai_specialization", "experience_level",
    "experience_years", "education_required", "industry", "company_size",
    "interview_rounds", "year", "work_mode", "weekly_hours", "company_rating",
    "job_openings", "hiring_difficulty_score", "layoff_risk",
    "ai_adoption_score", "company_funding_billion", "economic_index",
    "ai_maturity_years", "offer_acceptance_rate", "tax_rate_percent",
    "vacation_days", "skill_demand_score", "automation_risk",
    "job_security_score", "career_growth_score", "work_life_balance_score",
    "promotion_speed", "cost_of_living_index", "employee_satisfaction",
]

CATEGORICAL = {
    "country": ["Australia", "Brazil", "Canada", "France", "Germany", "India",
                "Japan", "Netherlands", "Singapore", "UAE", "UK", "USA"],
    "job_role": ["AI Engineer", "Computer Vision Engineer", "Data Analyst",
                 "Data Scientist", "Machine Learning Engineer", "NLP Engineer",
                 "Research Scientist", "Software Engineer AI"],
    "ai_specialization": ["Analytics", "Computer Vision", "Forecasting",
                          "Generative AI", "LLM", "MLOps", "NLP",
                          "Reinforcement Learning"],
    "industry": ["Automotive", "Consulting", "Education", "Energy", "Finance",
                 "Gaming", "Healthcare", "Retail", "Tech", "Telecom"],
    "work_mode": ["Hybrid", "Onsite", "Remote"],
}

ORDINAL = {
    "experience_level": ["Entry", "Mid", "Senior", "Lead"],
    "education_required": ["Bootcamp", "Diploma", "Bachelor", "Master", "PhD"],
    "company_size": ["Startup", "Small", "Medium", "Large", "Enterprise"],
}

CAT_DEFAULTS = {
    "country": "USA",
    "job_role": "Data Scientist",
    "ai_specialization": "Generative AI",
    "industry": "Tech",
    "work_mode": "Hybrid",
    "experience_level": "Mid",
    "education_required": "Bachelor",
    "company_size": "Medium",
}

# name, min, max, default, step
NUMERIC = [
    ("experience_years", 0, 19, 5, 1),
    ("interview_rounds", 2, 7, 4, 1),
    ("year", 2020, 2026, 2024, 1),
    ("weekly_hours", 36, 55, 40, 1),
    ("company_rating", 3.2, 4.8, 4.0, 0.1),
    ("job_openings", 1, 50, 5, 1),
    ("hiring_difficulty_score", 0, 100, 50, 1),
    ("layoff_risk", 0.0, 0.588, 0.2, 0.01),
    ("ai_adoption_score", 1, 100, 50, 1),
    ("company_funding_billion", 0.2, 9.5, 1.0, 0.1),
    ("economic_index", 45, 100, 70, 1),
    ("ai_maturity_years", 3, 14, 5, 1),
    ("offer_acceptance_rate", 55, 95, 75, 1),
    ("tax_rate_percent", 12, 42, 25, 1),
    ("vacation_days", 10, 30, 20, 1),
    ("skill_demand_score", 1, 100, 50, 1),
    ("automation_risk", 1, 100, 30, 1),
    ("job_security_score", 29, 99, 70, 1),
    ("career_growth_score", 25, 99, 70, 1),
    ("work_life_balance_score", 25, 98, 70, 1),
    ("promotion_speed", 12, 98, 50, 1),
    ("cost_of_living_index", 0.5, 2.5, 1.2, 0.1),
    ("employee_satisfaction", 42, 99, 70, 1),
]


@st.cache_resource
def load_salary_model():
    return joblib.load(MODEL_PKL)


model = load_salary_model()

st.title("Salary prediction")
st.caption("Darren's model: CatBoost regression on global AI and data jobs")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

with tab_single:
    st.markdown("Fill in the job details to predict the annual base salary in USD.")

    st.subheader("Job profile")
    cats = dict(CAT_DEFAULTS)
    for row in [["country", "job_role", "ai_specialization", "industry"],
                ["work_mode", "experience_level", "education_required", "company_size"]]:
        cols = st.columns(4)
        for c, name in zip(cols, row):
            options = CATEGORICAL.get(name, ORDINAL.get(name))
            cats[name] = c.selectbox(name.replace("_", " ").title(), options,
                                     index=options.index(CAT_DEFAULTS[name]))

    st.subheader("Numeric details")
    nums = {}
    for i in range(0, len(NUMERIC), 4):
        cols = st.columns(4)
        for c, spec in zip(cols, NUMERIC[i:i + 4]):
            name, lo, hi, default, step = spec
            nums[name] = c.number_input(
                name.replace("_", " ").title(), min_value=lo, max_value=hi,
                value=default, step=step,
            )

    if st.button("Predict salary", type="primary"):
        X = pd.DataFrame([{f: ({**cats, **nums})[f] for f in FEATURES}])
        pred = float(model.predict(X)[0])
        st.metric("Predicted annual salary", f"${pred:,.0f}")

with tab_batch:
    st.markdown(
        "Upload a CSV and predict salaries for all rows. The file must contain "
        "the model's 31 feature columns (the raw dataset or its subset works). "
        "Categorical values should use the same labels as the training data."
    )
    with st.expander("Download a sample input file"):
        sample = pd.read_csv(RAW_CSV)[FEATURES].head(200)
        buf = BytesIO()
        sample.to_csv(buf, index=False)
        st.download_button("Download sample CSV", buf.getvalue(), "salary_sample_input.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            X = df[FEATURES]
            pred = model.predict(X)
            result = pd.DataFrame(
                {f: X[f] for f in FEATURES}
            )
            result.insert(0, "predicted_salary_usd", pred.round(2))
            if "salary_usd" in df.columns:
                result.insert(1, "actual_salary_usd", df["salary_usd"])

            st.dataframe(result, width="stretch")
            st.metric("Rows predicted", len(result))
            buf = BytesIO()
            result.to_csv(buf, index=False)
            st.download_button("Download predictions", buf.getvalue(), "salary_predictions.csv", "text/csv")
