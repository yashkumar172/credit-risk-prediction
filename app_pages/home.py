"""Home page (SPEC §14.1): what the project does, the target, key limitations."""

import streamlit as st

from app_support import load_report

st.markdown(
    "This app estimates a borrower's **probability of default (PD)**. Here, *default* means **serious "
    "delinquency: being 90 days or more past due on a payment, or worse, at some point within the two years "
    "after the borrower's data was recorded**. It is not a legal default or a write-off. The model is a "
    "Random Forest trained on the Kaggle *Give Me Some Credit* dataset, with calibrated probabilities and "
    "SHAP explanations for every prediction."
)
st.warning("Educational and portfolio project. It is **not** suitable for real lending decisions.",
           icon=":material/gavel:")

test = load_report("test_metrics.json")
if test is not None:
    # Every number below is read from the saved one-time test evaluation (test_metrics.json)
    at = test["at_threshold"]
    ci = test["bootstrap"]["calibrated"]
    default_rate = test["test_split"]["default_rate"]
    st.subheader("Results on held-out test data")
    with st.container(horizontal=True):
        st.metric("PR-AUC", f"{test['calibrated']['pr_auc']:.3f}", border=True,
                  help="Primary metric. A model that ranks borrowers at random scores about "
                       f"{default_rate:.3f}, the default rate.")
        st.metric("ROC-AUC", f"{test['calibrated']['roc_auc']:.3f}", border=True)
        st.metric(f"Recall at {at['threshold']:.2f} threshold", f"{at['recall']:.1%}", border=True,
                  help=f"Share of borrowers who defaulted that were flagged (PD at or above {at['threshold']:.2f}).")
        st.metric("Borrowers flagged", f"{at['flagged_rate']:.1%}", border=True,
                  help=f"Share of borrowers flagged for review at threshold {at['threshold']:.2f}.")
    st.caption(f"{test['bootstrap']['level']:.0%} confidence intervals ({test['bootstrap']['resamples']:,} bootstrap "
               f"resamples of the test set): PR-AUC {ci['pr_auc']['lower']:.3f} to {ci['pr_auc']['upper']:.3f}, "
               f"ROC-AUC {ci['roc_auc']['lower']:.3f} to {ci['roc_auc']['upper']:.3f}.")
    st.markdown(
        f"**Why PR-AUC?** Only {default_rate:.1%} of borrowers default, so a model that predicts \"no default\" "
        f"for everyone would be {1 - default_rate:.0%} accurate and still useless. PR-AUC (precision-recall area, "
        "or average precision) looks only at how well the model finds the defaulters: it balances how many "
        "defaulters are caught (recall) against how many flagged borrowers really default (precision). Ranking "
        f"borrowers at random scores about {default_rate:.3f}, and a perfect model scores 1."
    )

st.subheader("How to use it")
st.markdown(
    "- **Prediction:** enter a borrower's details to get their PD, a risk band, whether they would be "
    "flagged for manual review, and the factors behind the estimate.\n"
    "- **Model performance:** test results with confidence intervals, curves and the model comparison.\n"
    "- **Explainability:** which features drive the model overall.\n"
    "- **About and limitations:** what the model cannot do, and responsible-use notes."
)

st.subheader("Key limitations")
st.markdown(
    "- The data is about 15 years old (2011-era), from an unknown population, with no loan details.\n"
    "- The PD estimates a historical pattern, not a cause. Shifts in the economy or the applicant "
    "population would make it less reliable.\n"
    "- The decision threshold and risk bands rest on illustrative cost assumptions, not banking standards.\n"
    "- The model uses age, a protected attribute in many lending regulations."
)
