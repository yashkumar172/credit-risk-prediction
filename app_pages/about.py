"""About and limitations page (SPEC §14.5, §21)."""

import streamlit as st

from app_support import load_report
from src import config

st.error("**Do not use this model for real lending decisions.** It is an educational project, trained on old "
         "public data, and has not been validated, audited or approved for any credit process.",
         icon=":material/gavel:")

st.subheader("Dataset limitations")
st.markdown(
    "- **Age of the data:** the Kaggle *Give Me Some Credit* data dates from around 2011. Credit behaviour, "
    "products and the economy have changed since.\n"
    "- **Unknown population:** the lender, country and selection of borrowers are not documented. The "
    "borrowers were already approved, so applicants who were rejected are missing (selection bias).\n"
    "- **No loan details:** loan amount, purpose, term and collateral are not available.\n"
    "- **Data quality:** income is missing for about 20% of borrowers, and some values are codes or errors "
    "(for example past-due counts of 96/98). These are treated as unknown, as documented in `DECISIONS.md`."
)

st.subheader("Responsible use")
shap = load_report("shap_global.json")
age_rank = next((f["rank"] for f in shap["features"] if f["feature"] == "age"), None) if shap else None
age_note = (f" It ranks {age_rank} of {len(shap['features'])} features by mean |SHAP|, and older borrowers "
            "receive lower PDs.") if age_rank else ""
st.markdown(
    "- **Historical bias:** the model learns from past lending outcomes, which can reflect past unequal "
    "treatment. It reproduces those patterns rather than correcting them.\n"
    "- **Age (protected attribute):** age is used because it is one of the dataset's source features. In many "
    f"jurisdictions, using age in credit decisions is restricted or prohibited.{age_note} A production model "
    "would need legal review, and possibly a version without age.\n"
    "- **Proxy variables:** even without protected attributes, features such as income, dependents or "
    "real-estate loans can act as proxies for them.\n"
    "- **Dataset shift:** the PD is reliable only for borrowers similar to those in the training data. The app "
    "warns when an input lies outside the training data's 1st-99th percentile range.\n"
    "- **Costs of errors:** the decision threshold assumes a missed defaulter costs "
    f"{config.COST_FALSE_NEGATIVE:g}× as much as a false alarm. These are illustrative numbers. At the chosen "
    "threshold most flagged borrowers do not default, which is the price of catching more defaulters.\n"
    "- **Interpretability:** SHAP shows which inputs moved a prediction. It describes the model's behaviour, "
    "not causes, and it explains the uncalibrated model.\n"
    "- **Human oversight:** the output is a **flag for manual review**, not a decision. A person should make "
    "and be accountable for any credit decision."
)

st.subheader("How the model was built")
st.markdown(
    "Stratified 70/15/15 train/validation/test split. Models were compared with 5-fold cross-validation on the "
    "training split, and a Random Forest was selected and tuned. It was calibrated (isotonic) and the threshold "
    "was chosen on the validation split. The test split was used once, for the final evaluation. Details, "
    "including every methodology decision, are in the project's `README.md` and `DECISIONS.md`."
)
