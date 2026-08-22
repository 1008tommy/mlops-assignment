from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Path to the saved PyCaret + CatBoost pipeline

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "darren"
    / "models"
    / "final_salary_catboost_pipeline.pkl"
)


# Features expected by the trained model

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


# Allowed categorical values

OPTIONS = {
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

    "experience_level": [
        "Entry", "Mid", "Senior", "Lead"
    ],

    "education_required": [
        "Bootcamp", "Diploma", "Bachelor", "Master", "PhD"
    ],

    "industry": [
        "Automotive", "Consulting", "Education", "Energy", "Finance",
        "Gaming", "Healthcare", "Retail", "Tech", "Telecom"
    ],

    "company_size": [
        "Startup", "Small", "Medium", "Large", "Enterprise"
    ],

    "work_mode": [
        "Hybrid", "Onsite", "Remote"
    ],
}


# Default values shown when the app starts

DEFAULTS = {
    "country": "USA",
    "job_role": "Data Scientist",
    "ai_specialization": "Generative AI",
    "experience_level": "Mid",
    "education_required": "Bachelor",
    "industry": "Tech",
    "company_size": "Medium",
    "work_mode": "Hybrid",
}


# Numeric input rules, so instead of just only loading from the data using its min and max, this set a more reasonable range for the user to input.
# Format: minimum, maximum, default, step

NUMERIC = {
    "experience_years": (0, 30, 5, 1),
    "interview_rounds": (1, 10, 4, 1),
    "year": (2020, 2026, 2026, 1),
    "weekly_hours": (10, 80, 40, 1),
    "company_rating": (1.0, 5.0, 4.0, 0.1),
    "job_openings": (1, 100, 5, 1),

    "hiring_difficulty_score": (0, 100, 50, 1),
    "layoff_risk": (0.0, 1.0, 0.2, 0.01),
    "ai_adoption_score": (0, 100, 50, 1),

    "company_funding_billion": (0.0, 50.0, 1.0, 0.1),
    "economic_index": (0, 100, 70, 1),
    "ai_maturity_years": (0, 20, 5, 1),

    "offer_acceptance_rate": (0, 100, 75, 1),
    "tax_rate_percent": (0, 60, 25, 1),
    "vacation_days": (0, 50, 20, 1),

    "skill_demand_score": (0, 100, 50, 1),
    "automation_risk": (0, 100, 30, 1),
    "job_security_score": (0, 100, 70, 1),
    "career_growth_score": (0, 100, 70, 1),
    "work_life_balance_score": (0, 100, 70, 1),
    "promotion_speed": (0, 100, 50, 1),

    "cost_of_living_index": (0.1, 5.0, 1.2, 0.1),
    "employee_satisfaction": (0, 100, 70, 1),
}


# Page configuration

st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="wide"
)


# Load the preprocessing pipeline and CatBoost model once

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# Convert feature names into readable labels

def display_name(name):
    return name.replace("_", " ").title()


# Validate input data
def validate_inputs(df, check_integer=False):
    df = df.copy()
    errors = []

    # Check categorical values
    for feature, allowed in OPTIONS.items():
        invalid = df.loc[~df[feature].isin(allowed), feature].unique()

        if len(invalid) > 0:
            errors.append(
                f"{display_name(feature)} has invalid value(s): "
                + ", ".join(map(str, invalid))
            )

    # Check numeric values
    for feature, settings in NUMERIC.items():
        minimum, maximum, _, step = settings
        converted = pd.to_numeric(df[feature], errors="coerce")

        if converted.isna().any():
            errors.append(
                f"{display_name(feature)} must be a valid number."
            )
            continue

        if check_integer and step == 1 and (converted % 1 != 0).any():
            errors.append(
                f"{display_name(feature)} must be a whole number."
            )
            continue

        outside = ~converted.between(minimum, maximum)

        if outside.any():
            bad_values = converted[outside].unique().tolist()
            errors.append(
                f"{display_name(feature)} must be between "
                f"{minimum} and {maximum}. "
                f"Invalid value(s): {bad_values}"
            )
            continue

        df[feature] = converted

    return df, errors


# Create one valid example row for the sample CSV

def create_sample():

    values = {
        **DEFAULTS,
        **{
            feature: settings[2]
            for feature, settings in NUMERIC.items()
        }
    }

    return pd.DataFrame([
        {
            feature: values[feature]
            for feature in FEATURES
        }
    ])


# Header

st.title("AI Job Salary Predictor")

st.caption(
    "Predict annual salaries for AI and data-related jobs "
    "using a trained CatBoost regression pipeline."
)


single_tab, batch_tab = st.tabs([
    "Single Prediction",
    "Batch Prediction"
])


# Single prediction

with single_tab:

    st.write(
        "Enter the job information below to predict "
        "the annual salary in USD."
    )

    inputs = {}


    # Categorical inputs

    st.subheader("Job Profile")

    categorical_items = list(
        OPTIONS.items()
    )

    for i in range(
        0,
        len(categorical_items),
        4
    ):

        cols = st.columns(4)

        for col, (
            feature,
            options
        ) in zip(
            cols,
            categorical_items[i:i + 4]
        ):

            with col:

                inputs[feature] = st.selectbox(
                    display_name(feature),
                    options,
                    index=options.index(
                        DEFAULTS[feature]
                    )
                )


    # Numeric inputs

    st.subheader("Additional Details")

    st.caption(
        "Enter a value within the accepted range shown for each field."
    )

    numeric_items = list(
        NUMERIC.items()
    )

    for i in range(
        0,
        len(numeric_items),
        4
    ):

        cols = st.columns(4)

        for col, (
            feature,
            settings
        ) in zip(
            cols,
            numeric_items[i:i + 4]
        ):

            minimum, maximum, default, _ = settings

            with col:

                # Text input allows invalid values to be entered
                # so the app can display a clear error message.
                inputs[feature] = st.text_input(
                    f"{display_name(feature)} "
                    f"({minimum} - {maximum})",
                    value=str(default),
                    key=f"single_{feature}"
                )


    # Generate single prediction

    if st.button(
        "Predict Salary",
        type="primary",
        use_container_width=True
    ):

        X = pd.DataFrame([
            {
                feature: inputs[feature]
                for feature in FEATURES
            }
        ])


        # Manual inputs use stricter whole-number validation
        X, errors = validate_inputs(
            X,
            check_integer=True
        )


        if errors:

            st.error(
                "Invalid input. Please correct the following:"
            )

            for error in errors:
                st.write(f"- {error}")


        else:

            try:

                # The saved pipeline automatically applies
                # preprocessing before generating the prediction.
                prediction = model.predict(X)

                predicted_salary = float(
                    prediction[0]
                )


                # Display the result neatly in the centre
                st.divider()

                left, centre, right = st.columns(
                    [1, 2, 1]
                )

                with centre:

                    st.markdown(
                        "<h3 style='text-align:center;'>"
                        "Prediction Result"
                        "</h3>",
                        unsafe_allow_html=True
                    )

                    st.success(
                        "Salary prediction completed successfully."
                    )

                    st.metric(
                        "Estimated Annual Salary (USD)",
                        f"${predicted_salary:,.0f}"
                    )


            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# Batch prediction

with batch_tab:

    st.write(
        "Upload a CSV containing the required model features "
        "to predict multiple salaries at once."
    )


    # Create downloadable sample CSV

    sample = create_sample()

    buffer = BytesIO()

    sample.to_csv(
        buffer,
        index=False
    )

    st.download_button(
        "Download Sample CSV",
        buffer.getvalue(),
        "salary_sample_input.csv",
        "text/csv"
    )


    # Upload CSV

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type="csv"
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


        # Preview uploaded data

        st.subheader("Uploaded Data")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.caption(
            f"{len(df):,} rows uploaded"
        )


        # Check that all required model features exist

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


        else:

            X = df[FEATURES].copy()


            # Batch data can contain valid decimal values,
            # so whole-number checking is disabled.
            X, errors = validate_inputs(
                X,
                check_integer=False
            )


            if errors:

                st.error(
                    "Invalid data found in the uploaded CSV:"
                )

                for error in errors:
                    st.write(f"- {error}")


            # Generate batch predictions

            elif st.button(
                "Run Batch Prediction",
                type="primary",
                use_container_width=True
            ):

                try:

                    predictions = model.predict(X)

                    result = X.copy()

                    result.insert(
                        0,
                        "predicted_salary_usd",
                        predictions.round(2)
                    )


                    # Compare predictions with actual salary
                    # when the uploaded file contains the target.
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


                    # Display batch prediction results

                    st.divider()

                    st.markdown(
                        "<h3 style='text-align:center;'>"
                        "Batch Prediction Results"
                        "</h3>",
                        unsafe_allow_html=True
                    )

                    st.success(
                        "Batch prediction completed successfully."
                    )


                    # Summary metrics

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Rows Predicted",
                        f"{len(result):,}"
                    )

                    col2.metric(
                        "Average Predicted Salary",
                        f"${predictions.mean():,.0f}"
                    )

                    if "absolute_error" in result.columns:

                        col3.metric(
                            "Average Absolute Error",
                            f"${result['absolute_error'].mean():,.0f}"
                        )


                    # Display prediction table

                    st.subheader(
                        "Prediction Results"
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )


                    # Allow prediction results to be downloaded

                    result_buffer = BytesIO()

                    result.to_csv(
                        result_buffer,
                        index=False
                    )

                    st.download_button(
                        "Download Predictions",
                        result_buffer.getvalue(),
                        "salary_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )


                except Exception as e:

                    st.error(
                        f"Batch prediction failed: {e}"
                    )