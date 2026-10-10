"""Explainability page (SPEC §14.4): global SHAP plots with the units note (SPEC §13.1)."""

import pandas as pd
import streamlit as st

from app_support import load_report, show_figure
from src.explain import PROBABILITY, SHAP_NOTES

report = load_report("shap_global.json")
if report is None:
    st.info("Global SHAP results have not been produced yet (`python -m src.explain`).", icon=":material/info:")
    st.stop()

units = report["units"]
st.info(SHAP_NOTES[units] + " The calibrator is monotonic, so it keeps the ranking and direction of each effect.",
        icon=":material/info:")
sample = report["sample"]
st.caption(f"Computed with {report['explainer']} on a stratified sample of {sample['rows']:,} validation borrowers "
           f"({sample['defaults']:,} defaults, seed {sample['random_state']}). The test set is not used.")

summary, importance = st.columns(2)
with summary:
    show_figure("shap_beeswarm.png",
                alt="SHAP summary plot: each dot is a borrower, coloured by feature value",
                caption="Each dot is one borrower. Right of zero raises the predicted risk; colour shows the "
                        "feature value (red high, blue low).")
with importance:
    show_figure("shap_mean_abs_bar.png", alt="Mean absolute SHAP value per feature",
                caption="Average size of each feature's contribution.")

scale = 100 if units == PROBABILITY else 1
unit = "pp" if units == PROBABILITY else "log-odds"
direction = {True: "Raises risk", False: "Lowers risk", None: "-"}
table = pd.DataFrame([{
    "Rank": f["rank"],
    "Feature": f["label"],
    f"Mean |SHAP| ({unit})": scale * f["mean_abs_shap"],
    "Higher value": direction[None if f["value_shap_spearman"] is None else f["value_shap_spearman"] > 0],
} for f in report["features"]])
st.subheader("Feature ranking")
st.dataframe(table, hide_index=True, alt="Features ranked by mean absolute SHAP value",
             column_config={f"Mean |SHAP| ({unit})": st.column_config.NumberColumn(format="%.2f")})
st.caption("\"Higher value\" is the sign of the rank correlation between a feature's value and its SHAP value "
           "in the sample. It describes the model, not causes.")
