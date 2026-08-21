from io import BytesIO
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# PATHS

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "darren"
    / "models"
    / "final_salary_catboost_pipeline.pkl"
)

SCHEMA_PATH = (
    ROOT
    / "darren"
    / "models"
    / "salary_model_schema.json"
)


# PAGE CONFIG

st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="wide"
)


# LOAD MODEL + SCHEMA

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r") as file:
        return json.load(file)


try:
    model = load_model()
    schema = load_schema()

except Exception as e:
    st.error(f"Unable to load model or schema: {e}")
    st.stop()


FEATURES = schema["features"]
CATEGORICAL = schema["categorical"]
ORDINAL = schema["ordinal"]
NUMERIC = schema["numeric"]


# HELPER FUNCTIONS

def display_name(column):
    return column.replace("_", " ").title()


def get_number_input(column, settings):
    minimum = settings["min"]
    maximum = settings["max"]
    default = settings["default"]

    # Use integer inputs when all values are whole numbers
    if (
        float(minimum).is_integer()
        and float(maximum).is_integer()
        and float(default).is_integer()
    ):
        return st.number_input(
            display_name(column),
            min_value=int(minimum),
            max_value=int(maximum),
            value=int(default),
            step=1,
        )

    return st.number_input(
        display_name(column),
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default),
        step=0.1,
    )


def make_sample_row():
    row = {}

    for column, options in CATEGORICAL.items():
        row[column] = options[0]

    for column, options in ORDINAL.items():
        row[column] = options[0]

    for column, settings in NUMERIC.items():
        row[column] = settings["default"]

    return {
        feature: row[feature]
        for feature in FEATURES
    }


# HEADER

st.title("AI Job Salary Predictor TEST")

st.caption(
    "Predict annual salary for AI and data-related jobs "
    "using the trained CatBoost regression pipeline."
)


# TABS


single_tab, batch_tab = st.tabs(
    ["Single Prediction", "Batch Prediction"]
)


# SINGLE PREDICTION

with single_tab:

    st.write(
        "Enter the job information below to estimate "
        "the annual salary in USD."
    )

    inputs = {}

    # Categorical + ordinal inputs

    with st.expander(
        "Job & Company Information",
        expanded=True
    ):

        categorical_inputs = {
            **CATEGORICAL,
            **ORDINAL,
        }

        items = list(
            categorical_inputs.items()
        )

        for i in range(0, len(items), 4):

            columns = st.columns(4)

            for container, (
                feature,
                options
            ) in zip(
                columns,
                items[i:i + 4]
            ):

                with container:
                    inputs[feature] = st.selectbox(
                        display_name(feature),
                        options,
                    )

    # Numeric inputs

    with st.expander(
        "Additional Job Details",
        expanded=False
    ):

        numeric_items = list(
            NUMERIC.items()
        )

        for i in range(
            0,
            len(numeric_items),
            4
        ):

            columns = st.columns(4)

            for container, (
                feature,
                settings
            ) in zip(
                columns,
                numeric_items[i:i + 4]
            ):

                with container:
                    inputs[feature] = (
                        get_number_input(
                            feature,
                            settings,
                        )
                    )

    # Predict

    if st.button(
        "Predict Salary",
        type="primary",
        use_container_width=True,
    ):

        try:

            X = pd.DataFrame(
                [
                    {
                        feature: inputs[feature]
                        for feature in FEATURES
                    }
                ]
            )

            prediction = model.predict(X)

            predicted_salary = float(
                prediction[0]
            )

            st.success(
                "Prediction completed successfully."
            )

            st.metric(
                "Predicted Annual Salary",
                f"${predicted_salary:,.0f}",
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# BATCH PREDICTION

with batch_tab:

    st.write(
        "Upload a CSV containing the required model "
        "input columns to predict multiple salaries at once."
    )

    # Sample CSV

    with st.expander(
        "Sample CSV Format"
    ):

        sample = pd.DataFrame(
            [make_sample_row()]
        )

        st.dataframe(
            sample,
            use_container_width=True,
        )

        sample_buffer = BytesIO()

        sample.to_csv(
            sample_buffer,
            index=False,
        )

        st.download_button(
            "Download Sample CSV",
            sample_buffer.getvalue(),
            "salary_sample_input.csv",
            "text/csv",
        )

    # Upload CSV

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
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
            use_container_width=True,
        )

        st.caption(
            f"{len(df):,} rows uploaded"
        )

        # Check required columns

        missing_columns = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

            st.stop()

        X = df[FEATURES].copy()
        
        # Missing value validation

        if X.isna().any().any():

            missing_value_columns = (
                X.columns[
                    X.isna().any()
                ].tolist()
            )

            st.error(
                "Missing values found in: "
                + ", ".join(
                    missing_value_columns
                )
            )

            st.stop()

        # Numeric validation

        invalid_numeric = []

        for feature in NUMERIC:

            converted = pd.to_numeric(
                X[feature],
                errors="coerce",
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
                + ", ".join(
                    invalid_numeric
                )
            )

            st.stop()


        # Category validation
        
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
                feature,
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
                "Invalid categorical values found."
            )

            for item in invalid_categories:
                st.write(f"- {item}")

            st.stop()

        # Batch prediction button

        if st.button(
            "Run Batch Prediction",
            type="primary",
            use_container_width=True,
        ):

            try:

                predictions = model.predict(
                    X
                )

                result = X.copy()

                result.insert(
                    0,
                    "predicted_salary_usd",
                    predictions.round(2),
                )

                # Optional actual salary comparison
                if "salary_usd" in df.columns:

                    actual = pd.to_numeric(
                        df["salary_usd"],
                        errors="coerce",
                    )

                    result.insert(
                        1,
                        "actual_salary_usd",
                        actual,
                    )

                    result.insert(
                        2,
                        "absolute_error",
                        (
                            actual
                            - predictions
                        ).abs().round(2),
                    )

                # Results

                st.success(
                    "Batch prediction completed successfully."
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Rows Predicted",
                    f"{len(result):,}",
                )

                col2.metric(
                    "Average Predicted Salary",
                    f"${predictions.mean():,.0f}",
                )

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    result,
                    use_container_width=True,
                )

                # Download results

                result_buffer = BytesIO()

                result.to_csv(
                    result_buffer,
                    index=False,
                )

                st.download_button(
                    "Download Predictions",
                    result_buffer.getvalue(),
                    "salary_predictions.csv",
                    "text/csv",
                )

            except Exception as e:

                st.error(
                    f"Batch prediction failed: {e}"
                )
