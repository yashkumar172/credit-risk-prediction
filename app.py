"""Streamlit app (SPEC §14): ``streamlit run app.py``.

The app only loads precomputed artifacts from ``models/`` and ``reports/``. It
never trains a model, never reads the raw dataset and never touches the test
set.
"""

import streamlit as st

st.set_page_config(page_title="Credit risk prediction", page_icon=":material/credit_score:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("app_pages/prediction.py", title="Prediction", icon=":material/calculate:"),
        st.Page("app_pages/performance.py", title="Model performance", icon=":material/monitoring:"),
        st.Page("app_pages/explainability.py", title="Explainability", icon=":material/psychology:"),
        st.Page("app_pages/about.py", title="About and limitations", icon=":material/info:"),
    ],
    position="top",
)

st.title(page.title, icon=page.icon)
page.run()
