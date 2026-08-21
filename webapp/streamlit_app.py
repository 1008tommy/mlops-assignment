"""IT3385 MLOps prediction web app (router).

Two pages: Javian's predictive maintenance classifier and Darren's salary
regressor. Run locally with:

    streamlit run webapp/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="IT3385 MLOps predictions", page_icon="🏭", layout="wide")

javian = st.Page(
    "views/javian_maintenance.py",
    title="Javian: Maintenance",
    icon="🏭",
    default=True,
)
darren = st.Page(
    "views/darren_salary.py",
    title="Darren: Salary"
)

nav = st.navigation([javian, darren])
nav.run()
