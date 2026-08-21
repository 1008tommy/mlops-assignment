from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# PATH

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "darren"
    / "models"
    / "final_salary_catboost_pipeline.pkl"
)


# MODEL INPUTS

FEATURES = [
    "country", "job_role", "ai_specialization", "experience_level",
    "experience_years", "education_required", "industry", "company_size",
    "interview_rounds", "year", "work_mode", "weekly_hours",
    "company_rating", "job_openings", "hiring_difficulty_score",
    "layoff_risk", "ai_adoption_score", "company_funding_billion",
    "economic_index", "ai_maturity_years", "offer_acceptance_rate",
    "tax_rate_percent", "vacation_days", "skill_demand_score",
    "automation_risk", "job_security_score", "career_growth_score",
    "work_life_balance_score", "promotion_speed",
    "cost_of_living_index", "employee_satisfaction",
]


CATEGORICAL = {
    "country": [
        "Australia", "Brazil", "Canada", "France", "Germany", "India",
        "Japan", "Netherlands", "Singapore", "UAE", "UK", "USA"
    ],

    "job_role": [
        "AI Engineer", "Computer Vision Engineer", "Data Analyst",
        "Data Scientist", "Machine Learning Engineer", "NLP Engineer",
        "Research Scientist", "Software Engineer AI"
    ],

    "ai_specialization": [
        "Analytics", "Computer Vision", "Forecasting", "Generative AI",
        "LLM", "MLOps", "NLP", "Reinforcement Learning"
    ],

    "industry": [
        "Automotive", "Consulting", "Education", "Energy", "Finance",
        "Gaming", "Healthcare", "Retail", "Tech", "Telecom"
    ],

    "work_mode": [
        "Hybrid", "Onsite", "Remote"
    ],
}


ORDINAL = {
    "experience_level": [
        "Entry", "Mid", "Senior", "Lead"
    ],

    "education_required": [
        "Bootcamp", "Diploma", "Bachelor", "Master", "PhD"
    ],

    "company_size": [
        "Startup", "Small", "Medium", "Large", "Enterprise"
    ],
}


DEFAULTS = {
    "country": "USA",
    "job_role": "Data Scientist",
    "ai_specialization": "Generative AI",
    "industry": "Tech",
    "work_mode": "Hybrid",
    "experience_level": "Mid",
    "education_required": "Bachelor",
    "company_size": "Medium",
}


# name: (minimum, maximum, default, step)

NUMERIC = {
    "experience_years": (0, 19, 5, 1),
    "interview_rounds": (2, 7, 4, 1),
    "year": (2020, 2026, 2024, 1),
    "weekly_hours": (36, 55, 40, 1),
    "company_rating": (3.2, 4.8, 4.0, 0.1),
    "job_openings": (1, 50, 5, 1),
    "hiring_difficulty_score": (0, 100, 50, 1),
    "layoff_risk": (0.0, 0.588, 0.2, 0.01),
    "ai_adoption_score": (1, 100, 50, 1),
    "company_funding_billion": (0.2, 9.5, 1.0, 0.1),
    "economic_index": (45, 100, 70, 1),
    "ai_maturity_years": (3, 14, 5, 1),
    "offer_acceptance_rate": (55, 95, 75, 1),
    "tax_rate_percent": (12, 42, 25, 1),
    "vacation_days": (10, 30, 20, 1),
    "skill_demand_score": (1, 100, 50, 1),
    "automation_risk": (1, 100, 30, 1),
    "job_security_score": (29, 99, 70, 1),
    "career_growth_score": (25, 99, 70, 1),
    "work_life_balance_score": (25, 98, 70, 1),
    "promotion_speed": (12, 98, 50, 1),
    "cost_of_living_index": (0.5, 2.5, 1.2, 0.1),
    "employee_satisfaction": (42, 99, 70, 1),
}


# PAGE CONFIG

st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="wide"
)


# LOAD MODEL

@st.cache_resource
def load_salary_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_salary_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# HELPERS

def display_name(name):
    return name.replace("_", " ").title()


def create_sample():
    values = dict(DEFAULTS)

    for name, (_, _, default, _) in NUMERIC.items():
        values[name] = default

    return pd.DataFrame([
        {
            feature: values[feature]
            for feature in FEATURES
        }
    ])

# HEADER

st.title("AI Job Salary Predictor")

st.caption(
    "Predict annual salaries for AI and data-related jobs "
    "using a trained CatBoost regression pipeline."
)


single_tab, batch_tab = st.tabs([
    "Single Prediction",
    "Batch Prediction"
])

# SINGLE PREDICTION

with single_tab:

    st.write(
        "Enter the job information below to predict "
        "the annual salary in USD."
    )

    inputs = {}

    st.subheader("Job Profile")

    categorical_inputs = {
        **CATEGORICAL,
        **ORDINAL,
    }

    items = list(categorical_inputs.items())

    for i in range(0, len(items), 4):

        cols = st.columns(4)

        for col, (feature, options) in zip(
            cols,
            items[i:i + 4]
        ):

            with col:

                inputs[feature] = st.selectbox(
                    display_name(feature),
                    options,
                    index=options.index(
                        DEFAULTS[feature]
                    ),
                )


    st.subheader("Additional Details")

    numeric_items = list(NUMERIC.items())

    for i in range(0, len(numeric_items), 4):

        cols = st.columns(4)

        for col, (
            feature,
            settings
        ) in zip(
            cols,
            numeric_items[i:i + 4]
        ):

            minimum, maximum, default, step = settings

            with col:

                inputs[feature] = st.number_input(
                    display_name(feature),
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                )


    if st.button(
        "Predict Salary",
        type="primary",
        use_container_width=True
    ):

        try:

            X = pd.DataFrame([
                {
                    feature: inputs[feature]
                    for feature in FEATURES
                }
            ])

            prediction = model.predict(X)

            predicted_salary = float(
                prediction[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Predicted Annual Salary",
                f"${predicted_salary:,.0f}"
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

# BATCH PREDICTION

with batch_tab:

    st.write(
        "Upload a CSV containing the required model input "
        "features to predict multiple salaries at once."
    )
    
    # SAMPLE CSV

    sample = create_sample()

    sample_buffer = BytesIO()

    sample.to_csv(
        sample_buffer,
        index=False
    )

    st.download_button(
        "Download Sample CSV",
        sample_buffer.getvalue(),
        "salary_sample_input.csv",
        "text/csv"
    )

    # UPLOAD CSV
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as e:

            st.error(
                f"Unable to read CSV: {e}"
            )

            st.stop()


        st.subheader("Uploaded Data")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.caption(
            f"{len(df):,} rows uploaded"
        )

        # CHECK COLUMNS

        missing_columns = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

            st.stop()


        X = df[FEATURES].copy()

        # CHECK MISSING VALUES

        if X.isna().any().any():

            missing_cols = (
                X.columns[
                    X.isna().any()
                ].tolist()
            )

            st.error(
                "Missing values found in: "
                + ", ".join(missing_cols)
            )

            st.stop()

        # NUMERIC VALIDATION

        invalid_numeric = []

        for feature in NUMERIC:

            converted = pd.to_numeric(
                X[feature],
                errors="coerce"
            )

            if converted.isna().any():

                invalid_numeric.append(
                    feature
                )

            else:

                X[feature] = converted


        if invalid_numeric:

            st.error(
                "Invalid numeric values found in: "
                + ", ".join(invalid_numeric)
            )

            st.stop()


        # CATEGORY VALIDATION

        allowed_categories = {
            **CATEGORICAL,
            **ORDINAL,
        }

        invalid_categories = []


        for feature, allowed in (
            allowed_categories.items()
        ):

            invalid = X.loc[
                ~X[feature].isin(allowed),
                feature
            ].unique()

            if len(invalid) > 0:

                invalid_categories.append(
                    f"{display_name(feature)}: "
                    + ", ".join(
                        map(str, invalid)
                    )
                )


        if invalid_categories:

            st.error(
                "Invalid categorical values found:"
            )

            for error in invalid_categories:
                st.write(f"- {error}")

            st.stop()


        # BATCH PREDICTION

        if st.button(
            "Run Batch Prediction",
            type="primary",
            use_container_width=True
        ):

            try:

                predictions = model.predict(
                    X
                )

                result = X.copy()

                result.insert(
                    0,
                    "predicted_salary_usd",
                    predictions.round(2)
                )


                # If actual salary exists,
                # show prediction error
                if "salary_usd" in df.columns:

                    actual = pd.to_numeric(
                        df["salary_usd"],
                        errors="coerce"
                    )

                    result.insert(
                        1,
                        "actual_salary_usd",
                        actual
                    )

                    result.insert(
                        2,
                        "absolute_error",
                        (
                            actual
                            - predictions
                        ).abs().round(2)
                    )


                st.success(
                    "Batch prediction completed successfully."
                )


                col1, col2 = st.columns(2)

                col1.metric(
                    "Rows Predicted",
                    f"{len(result):,}"
                )

                col2.metric(
                    "Average Predicted Salary",
                    f"${predictions.mean():,.0f}"
                )


                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                # DOWNLOAD RESULTS

                result_buffer = BytesIO()

                result.to_csv(
                    result_buffer,
                    index=False
                )

                st.download_button(
                    "Download Predictions",
                    result_buffer.getvalue(),
                    "salary_predictions.csv",
                    "text/csv"
                )


            except Exception as e:

                st.error(
                    f"Batch prediction failed: {e}"
                )
