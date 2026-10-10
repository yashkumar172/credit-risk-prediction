"""Model performance page (SPEC §14.3): saved reports and figures only."""

import pandas as pd
import streamlit as st

from app_support import load_report, show_figure

MODEL_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
IMBALANCE_NAMES = {"none": "None", "class_weight": "Class weights", "scale_pos_weight": "scale_pos_weight"}
METRIC_NAMES = {"pr_auc": "PR-AUC (primary)", "roc_auc": "ROC-AUC", "brier": "Brier score"}

comparison = load_report("model_comparison.json")
test = load_report("test_metrics.json")

# --- final test results ----------------------------------------------------------------------
st.subheader("Final test results")
if test is None:
    st.info("The one-time test evaluation has not been run yet (`python -m src.evaluate --final`).",
            icon=":material/info:")
else:
    st.caption(f"{test['test_split']['rows']:,} held-out borrowers ({test['test_split']['defaults']:,} defaults), "
               "evaluated once after every modelling choice was fixed. Ranges are 95% bootstrap confidence "
               f"intervals ({test['bootstrap']['resamples']:,} resamples).")
    at = test["at_threshold"]
    with st.container(horizontal=True):
        st.metric("PR-AUC", f"{test['calibrated']['pr_auc']:.3f}", border=True,
                  help="Primary metric: average precision of the deployed (calibrated) model.")
        st.metric("ROC-AUC", f"{test['calibrated']['roc_auc']:.3f}", border=True)
        st.metric("Precision", f"{at['precision']:.1%}", border=True,
                  help=f"Share of flagged borrowers who defaulted, at threshold {at['threshold']:.2f}.")
        st.metric("Recall", f"{at['recall']:.1%}", border=True,
                  help=f"Share of defaulters who were flagged, at threshold {at['threshold']:.2f}.")
        st.metric("F1", f"{at['f1']:.3f}", border=True)

    rows = []
    for metric, label in METRIC_NAMES.items():
        row = {"Metric": label}
        for model, column in (("calibrated", "Deployed (calibrated)"), ("raw", "Raw Random Forest")):
            ci = test["bootstrap"][model][metric]
            row[column] = f"{test[model][metric]:.4f}  [{ci['lower']:.4f}, {ci['upper']:.4f}]"
        rows.append(row)
    rows.append({"Metric": "Mean predicted PD",
                 "Deployed (calibrated)": f"{test['calibrated']['mean_predicted']:.2%}",
                 "Raw Random Forest": f"{test['raw']['mean_predicted']:.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 alt="Test metrics with 95% confidence intervals for the deployed and raw models")
    st.caption(f"Observed default rate on test: {test['test_split']['default_rate']:.2%}.")

    roc_pr, confusion, calibration, threshold = st.tabs(
        ["ROC and PR curves", "Confusion matrix", "Calibration", "Threshold analysis"])
    with roc_pr:
        show_figure("test_roc_pr_curves.png", alt="ROC and precision-recall curves on the test split")
    with confusion:
        show_figure("test_confusion_matrix.png",
                    alt=f"Confusion matrix on the test split at threshold {at['threshold']:.2f}")
    with calibration:
        show_figure("test_calibration_curve.png",
                    alt="Reliability diagram: predicted PD against observed default rate, raw and calibrated")
        bands = pd.DataFrame(test["risk_bands"]).rename(columns={
            "band": "Risk band", "rows": "Borrowers", "share": "Share", "mean_predicted": "Mean predicted PD",
            "observed_rate": "Observed default rate", "defaults": "Defaults"})
        st.dataframe(bands, hide_index=True, alt="Predicted and observed default rates per risk band on test",
                     column_config={c: st.column_config.NumberColumn(format="percent")
                                    for c in ("Share", "Mean predicted PD", "Observed default rate")})
    with threshold:
        show_figure("threshold_analysis_validation.png",
                    alt="Precision, recall and expected cost across decision thresholds on the validation split")
        st.caption("The threshold was chosen on the validation split, never on test.")

# --- model comparison ------------------------------------------------------------------------
st.subheader("Model comparison (development)")
if comparison is None:
    st.info("Run `python -m src.train` to produce the model comparison.", icon=":material/info:")
else:
    selection = comparison["selection"]
    # The deployed model: the tuned candidate if tuning was kept, else the selected untuned one
    deployed = next((c["candidate"] for c in comparison["candidates"] if c.get("stage") == "tuned"), None) \
        if selection.get("final_configuration") == "tuned" else selection["selected_candidate"]
    table = pd.DataFrame([{
        "Model": MODEL_NAMES[c["model"]] + (" (tuned)" if c.get("stage") == "tuned" else ""),
        "Imbalance handling": IMBALANCE_NAMES[c["imbalance"]],
        "Deployed": c["candidate"] == deployed,
        "CV PR-AUC": c["cv"]["pr_auc_mean"],
        "CV std": c["cv"]["pr_auc_std"],
        "Val PR-AUC": c["validation"]["pr_auc"],
        "Val ROC-AUC": c["validation"]["roc_auc"],
        "Val Brier": c["validation"]["brier"],
        "Mean PD": c["validation"]["mean_predicted"],
        "Fit time (s)": c["fit_time_s"],
    } for c in comparison["candidates"]])
    st.dataframe(table, hide_index=True, alt="Cross-validation and validation metrics for every candidate model",
                 column_config={
                     **{c: st.column_config.NumberColumn(format="%.4f")
                        for c in ("CV PR-AUC", "CV std", "Val PR-AUC", "Val ROC-AUC", "Val Brier")},
                     "Mean PD": st.column_config.NumberColumn(format="percent"),
                     "Fit time (s)": st.column_config.NumberColumn(format="%.1f"),
                 })
    st.caption(f"Cross-validation: stratified {comparison['cv_folds']}-fold on the training split. Val: the "
               "validation split. Selection: highest validation PR-AUC. "
               f"{comparison['selection']['reason'].capitalize()}.")
