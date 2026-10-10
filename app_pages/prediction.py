"""Prediction page (SPEC §14.2): raw inputs → calibrated PD, risk band, review flag and
local SHAP explanation (SPEC §13.3)."""

import altair as alt
import pandas as pd
import streamlit as st

from app_support import predictor_or_stop
from src import config
from src.explain import PROBABILITY, SHAP_NOTES_ONE_SENTENCE, format_contribution
from src.evaluate import BLUE, ORANGE
from src.predict import INPUT_SPECS, is_unknown

predictor = predictor_or_stop()

st.caption("Enter a borrower's details to estimate the probability of serious delinquency (90+ days past due) "
           "within two years. Educational project: not for real lending decisions.")
for message in predictor.version_warnings:
    st.warning(message, icon=":material/warning:")

# --- inputs: built from the metadata's raw feature list, in its order (SPEC §14.2) -----------
values = {}
with st.container(border=True):
    st.subheader("Borrower details")
    columns = st.columns(2)
    half = (len(predictor.raw_features) + 1) // 2  # first half left, second half right: reading order = metadata order
    for i, name in enumerate(predictor.raw_features):
        spec = INPUT_SPECS[name]
        label = config.FEATURE_LABELS[name]
        unknown_key = f"{name}__unknown"
        unknown = st.session_state.get(unknown_key, False)
        # The model ignores the debt ratio when income is unknown (DECISIONS.md, Phase 4)
        income_unknown = name == "DebtRatio" and st.session_state.get("MonthlyIncome__unknown", False)
        with columns[0 if i < half else 1]:
            number = st.number_input(
                label,
                min_value=int(spec.min_value) if spec.integer else spec.min_value,
                max_value=(int(spec.max_value) if spec.integer else spec.max_value) if spec.max_value else None,
                value=int(spec.default) if spec.integer else float(spec.default),
                step=int(spec.step) if spec.integer else float(spec.step),
                format=None if spec.integer else ("%.0f" if spec.step >= 1 else "%.2f"),
                help=spec.help,
                key=name,
                disabled=unknown or income_unknown,
            )
            if spec.allow_unknown:
                st.checkbox("Unknown", key=unknown_key, help=f"Send {label.lower()} as missing; the model handles it.")
            if income_unknown:
                st.caption("Not used: the model ignores the debt ratio when income is unknown.")
            values[name] = float("nan") if unknown else number

    predict_clicked = st.button("Predict risk", type="primary", icon=":material/query_stats:")

snapshot = {name: None if is_unknown(value) else value for name, value in values.items()}
if predict_clicked:
    try:
        st.session_state.prediction = (snapshot, predictor.predict(values))
    except ValueError as error:
        st.session_state.pop("prediction", None)
        st.error(f"Please correct the inputs: {error}", icon=":material/error:")

stored = st.session_state.get("prediction")
if stored is None:
    st.info("Enter the borrower's details and select **Predict risk**.", icon=":material/info:")
    st.stop()
if stored[0] != snapshot:
    st.info("The inputs have changed. Select **Predict risk** to update the result.", icon=":material/info:")
    st.stop()

# --- result (SPEC §14.2) ---------------------------------------------------------------------
prediction = stored[1]
explanation = prediction.explanation
with st.container(horizontal=True):
    st.metric("Probability of default", f"{prediction.pd:.1%}", border=True)
    st.metric("Risk band", prediction.risk_band, border=True,
              help=f"LOW below {config.RISK_BAND_LOW_UPPER:.0%}, MEDIUM below {config.RISK_BAND_MEDIUM_UPPER:.0%}, "
                   "HIGH otherwise. A descriptive aid with illustrative boundaries, not a decision.")
    st.metric("Flagged for manual review", "Yes" if prediction.flagged else "No", border=True,
              help=f"Flagged when the PD is at or above the decision threshold ({prediction.threshold:.2f}).")
st.caption(f"Decision threshold: {prediction.threshold:.2f}, chosen on validation data to minimise "
           f"{config.COST_FALSE_NEGATIVE:g} × missed defaults + {config.COST_FALSE_POSITIVE:g} × false alarms. "
           "These costs are illustrative assumptions.")
for message in prediction.warnings:
    st.warning(message, icon=":material/warning:")

# --- local explanation (SPEC §13.3) ----------------------------------------------------------
st.subheader("What drives this prediction")
increasing, decreasing = st.columns(2)
for column, title, factors in ((increasing, "Factors increasing risk", explanation.increasing),
                               (decreasing, "Factors reducing risk", explanation.decreasing)):
    with column, st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {f.label}: {f.display_value} ({format_contribution(f.contribution, explanation.units)})"
                              for f in factors) or "None")

scale = 100 if explanation.units == PROBABILITY else 1
unit = "percentage points of the uncalibrated PD" if explanation.units == PROBABILITY else "log-odds"
shown = sorted(explanation.factors, key=lambda f: abs(f.contribution), reverse=True)[:10]
chart_data = pd.DataFrame({
    "Factor": [f"{f.label} = {f.display_value}" for f in shown],
    "Contribution": [scale * f.contribution for f in shown],
    "Effect": ["Raises risk" if f.contribution > 0 else "Lowers risk" for f in shown],
})
chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X("Contribution:Q", title=f"SHAP contribution ({unit})"),
    y=alt.Y("Factor:N", sort=None, title=None),
    color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[ORANGE, BLUE]),
                    legend=alt.Legend(title=None, orient="bottom")),
    tooltip=["Factor", alt.Tooltip("Contribution:Q", format="+.2f"), "Effect"],
)
st.altair_chart(chart, alt="Bar chart of the ten largest SHAP contributions to this borrower's predicted risk")

st.caption(SHAP_NOTES_ONE_SENTENCE[explanation.units])
if explanation.units == PROBABILITY:
    st.caption(f"Average uncalibrated PD {explanation.base_value:.1%} + contributions = uncalibrated PD "
               f"{explanation.raw_output:.1%}, which the calibrator maps to the {prediction.pd:.1%} shown.")
