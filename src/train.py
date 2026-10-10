"""Training entry point: ``python -m src.train`` (SPEC §15.2).

Stages implemented so far:
    Phase 6 - Logistic Regression baseline
    Phase 7 - Random Forest and XGBoost, the imbalance-strategy comparison (SPEC §5.3)
              and model selection (SPEC §12)
    Phase 8 - Hyperparameter tuning of the selected model (SPEC §9), written to
              ``reports/metrics/tuning_results.json``. The search takes ~26 min on a
              6-core laptop, so an identical previous search is reused; pass
              ``--retune`` to force a new one.
    Phase 9 - Calibration (SPEC §11) and decision threshold (SPEC §10.2) on the
              validation split; saves the single artifact
              ``models/credit_risk_pipeline.joblib`` and ``models/model_metadata.json``.
              ``--from-phase 9`` runs only this stage: the saved Phase 6-8 results
              are reused (and left unchanged), and the final configuration is
              refitted on train and checked against its saved validation PR-AUC.
    Phase 10 - Global SHAP explanation (SPEC §13.2) on a stratified validation sample,
              via ``src.explain`` (also runnable alone: ``python -m src.explain``).

Every candidate is evaluated with stratified 5-fold CV on the training split,
then fitted on the full training split and scored on the validation split.
The test split is never used here. Results are written to
``reports/metrics/model_comparison.json``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import logging
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.calibration import (
    ISOTONIC_MIN_RELATIVE_GAIN,
    calibrated_brier_cv,
    choose_calibration_method,
    compare_calibration_methods,
    fit_calibrator,
)
from src.data_loader import DataSplit, file_sha256, load_raw_data, split_data
from src.evaluate import (
    DISPLAY_THRESHOLDS,
    choose_threshold,
    metrics_at_threshold,
    plot_threshold_analysis,
    probability_metrics,
)
from src.explain import run_global_explanation
from src.feature_engineering import CleaningTransformer
from src.preprocessing import make_model_pipeline
from src.tune import N_ITER, search_signature, tune_model

logger = logging.getLogger(__name__)

CV_SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"}

# Imbalance strategies compared per model (SPEC §5.3). SMOTE is optional and not run.
IMBALANCE_STRATEGIES = {
    "logistic_regression": ("none", "class_weight"),
    "random_forest": ("none", "class_weight"),
    "xgboost": ("none", "scale_pos_weight"),
}
SIMPLICITY_RANK = {"logistic_regression": 0, "random_forest": 1, "xgboost": 2}  # tie-break: simpler first
TIE_TOLERANCE = 0.005  # PR-AUC differences within this count as ties (SPEC §12)


def make_estimator(model_name: str, imbalance: str = "none", scale_pos_weight: float = 1.0, n_jobs: int = -1):
    """Untuned (default or lightly configured) estimator for SPEC §12 model selection."""
    weighted = imbalance != "none"
    if imbalance not in IMBALANCE_STRATEGIES.get(model_name, ()):
        raise ValueError(f"Unknown model/imbalance combination: {model_name}/{imbalance}")
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced" if weighted else None,
                                  random_state=config.RANDOM_STATE)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                      class_weight="balanced" if weighted else None,
                                      n_jobs=n_jobs, random_state=config.RANDOM_STATE)
    return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist", device="cpu",
                         scale_pos_weight=scale_pos_weight if weighted else 1.0,
                         eval_metric="logloss", n_jobs=n_jobs, random_state=config.RANDOM_STATE)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=config.TARGET), df[config.TARGET]


def cross_validate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                            n_folds: int = config.CV_FOLDS, n_jobs: int = -1) -> dict[str, float]:
    """Stratified k-fold CV. The whole pipeline is refitted on each training fold."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # fail loudly instead of reporting an unconverged model
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=CV_SCORING, n_jobs=n_jobs, error_score="raise")
    result = {}
    for name in CV_SCORING:
        values = scores[f"test_{name}"] * (-1 if name == "brier" else 1)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
    result["pr_auc_folds"] = [float(v) for v in scores["test_pr_auc"]]
    return result


def positive_weight(y_train: pd.Series) -> float:
    """Negative/positive ratio from the training split. Stratified folds keep the
    same ratio, so using it inside CV leaks nothing about individual rows."""
    return float((y_train == 0).sum() / (y_train == 1).sum())


def evaluate_candidate(model_name: str, imbalance: str, split: DataSplit, params: dict | None = None,
                       cv_result: dict | None = None, label: str | None = None) -> tuple[Pipeline, dict]:
    """CV on train, fit on the full training split, score raw and calibrated on validation.

    ``params`` override the untuned estimator settings (e.g. tuned values).
    ``cv_result`` reuses CV scores already computed on the same folds.
    """
    X_train, y_train = split_xy(split.train)
    X_val, y_val = split_xy(split.validation)

    ratio = positive_weight(y_train)
    estimator = make_estimator(model_name, imbalance, scale_pos_weight=ratio).set_params(**(params or {}))
    pipeline = make_model_pipeline(model_name, estimator)
    if cv_result is None:
        # LR is single-threaded, so parallelise its folds; tree models already use every core.
        cv_result = cross_validate_pipeline(pipeline, X_train, y_train,
                                            n_jobs=-1 if model_name == "logistic_regression" else 1)

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    predict_time = time.perf_counter() - start

    result = {
        "candidate": label or f"{model_name}[{imbalance}]",
        "model": model_name,
        "imbalance": imbalance,
        "cv": cv_result,
        "validation": probability_metrics(y_val, val_proba),
        "validation_brier_calibrated": calibrated_brier_cv(pipeline, X_val, y_val),
        "fit_time_s": round(fit_time, 3),
        "predict_time_s": round(predict_time, 3),
    }
    if params:
        result["params"] = params
    if imbalance == "scale_pos_weight":
        result["scale_pos_weight"] = round(ratio, 4)
    if model_name == "logistic_regression":
        result["coefficients"] = logistic_coefficients(pipeline)

    v = result["validation"]
    logger.info(
        "%-34s CV PR-AUC %.4f ± %.4f | val PR-AUC %.4f | ROC-AUC %.4f | Brier %.4f (cal. %.4f) | mean PD %.4f | fit %.1fs",
        result["candidate"], cv_result["pr_auc_mean"], cv_result["pr_auc_std"], v["pr_auc"], v["roc_auc"],
        v["brier"], result["validation_brier_calibrated"], v["mean_predicted"], fit_time,
    )
    return pipeline, result


def logistic_coefficients(pipeline: Pipeline) -> list[dict]:
    """Coefficients of a fitted LR pipeline, largest effect first (on preprocessed features)."""
    names = pipeline[:-1].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_.ravel()
    order = np.argsort(-np.abs(coefs))
    return [{"feature": str(names[i]), "coefficient": round(float(coefs[i]), 4)} for i in order]


def run_candidates(split: DataSplit, models=config.MODEL_NAMES) -> list[dict]:
    """Evaluate every model with every imbalance strategy."""
    return [evaluate_candidate(model, imbalance, split)[1]
            for model in models for imbalance in IMBALANCE_STRATEGIES[model]]


def choose_imbalance(candidates: list[dict]) -> dict[str, dict]:
    """SPEC §5.3: per model, the strategy with the best CV PR-AUC. Within the tie
    tolerance, no reweighting wins (simpler, and it does not distort probabilities)."""
    chosen = {}
    for model in dict.fromkeys(c["model"] for c in candidates):
        options = [c for c in candidates if c["model"] == model]
        best = max(c["cv"]["pr_auc_mean"] for c in options)
        tied = [c for c in options if best - c["cv"]["pr_auc_mean"] <= TIE_TOLERANCE]
        chosen[model] = next((c for c in tied if c["imbalance"] == "none"),
                             max(tied, key=lambda c: c["cv"]["pr_auc_mean"]))
    return chosen


def select_model(candidates: list[dict]) -> dict:
    """SPEC §12: highest validation PR-AUC; models within 0.005 are tied and broken by
    lower calibrated Brier score, then simplicity, then fit time."""
    finalists = choose_imbalance(candidates)
    best = max(c["validation"]["pr_auc"] for c in finalists.values())
    tied = [c for c in finalists.values() if best - c["validation"]["pr_auc"] <= TIE_TOLERANCE]
    winner = min(tied, key=lambda c: (c["validation_brier_calibrated"], SIMPLICITY_RANK[c["model"]], c["fit_time_s"]))

    if len(tied) == 1:
        reason = f"highest validation PR-AUC ({winner['validation']['pr_auc']:.4f}); no other model within {TIE_TOLERANCE}"
    else:
        reason = (f"tied on validation PR-AUC (within {TIE_TOLERANCE}) with "
                  f"{[c['candidate'] for c in tied if c is not winner]}; lowest calibrated Brier "
                  f"({winner['validation_brier_calibrated']:.5f})")
    return {
        "imbalance_choice": {m: c["imbalance"] for m, c in finalists.items()},
        "finalists": [c["candidate"] for c in sorted(finalists.values(), key=lambda c: -c["validation"]["pr_auc"])],
        "tied_with_best": [c["candidate"] for c in tied],
        "selected_model": winner["model"],
        "selected_imbalance": winner["imbalance"],
        "selected_candidate": winner["candidate"],
        "reason": reason,
    }


def compressed_size_mb(obj) -> float:
    """Size of ``obj`` saved with joblib (compress=3), as the final artifact will be."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer, compress=3)
    return round(buffer.tell() / 1e6, 1)


def load_cached_tuning(path, signature: dict) -> dict | None:
    """Previous search results, if they were produced by an identical search."""
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    return cached if cached.get("signature") == signature else None


def run_tuning(selection: dict, candidates: list[dict], split: DataSplit, data_sha256: str,
               retune: bool = False, cache_path=config.TUNING_RESULTS_PATH) -> tuple[dict, dict, Pipeline]:
    """SPEC §9: tune the selected model on the training split, then keep the tuned
    configuration only if its CV PR-AUC beats the untuned one on the same folds.

    The search (the slow part) is reused from ``cache_path`` when its signature
    matches exactly, unless ``retune`` is set. Everything after the search is
    always recomputed. Returns the tuning report, the tuned candidate's results
    and the final pipeline (fitted on the training split).
    """
    model, imbalance = selection["selected_model"], selection["selected_imbalance"]
    X_train, y_train = split_xy(split.train)
    signature = search_signature(model, imbalance, data_sha256)

    tuning = None if retune else load_cached_tuning(cache_path, signature)
    if tuning is not None:
        logger.info("Reusing cached search from %s (identical signature; pass --retune to search again)",
                    cache_path.name)
        tuning["reused_cached_search"] = True
    else:
        estimator = make_estimator(model, imbalance, scale_pos_weight=positive_weight(y_train), n_jobs=1)
        logger.info("Tuning %s (%d iterations x %d folds)...", model, N_ITER, config.CV_FOLDS)
        tuning = tune_model(model, estimator, X_train, y_train)
        tuning["signature"] = signature
        tuning["reused_cached_search"] = False

    untuned = next(c for c in candidates if c["candidate"] == selection["selected_candidate"])
    improved = tuning["best_cv_pr_auc_mean"] > untuned["cv"]["pr_auc_mean"]
    tuning["untuned_cv_pr_auc_mean"] = untuned["cv"]["pr_auc_mean"]
    tuning["final_configuration"] = "tuned" if improved else "untuned"
    logger.info("Best CV PR-AUC %.4f ± %.4f (untuned %.4f; search time %.0fs) -> keeping the %s configuration",
                tuning["best_cv_pr_auc_mean"], tuning["best_cv_pr_auc_std"], untuned["cv"]["pr_auc_mean"],
                tuning["tuning_time_s"], tuning["final_configuration"])

    best_cv = next(t for t in tuning["trials"] if t["params"] == tuning["best_params"])
    cv_result = {"pr_auc_mean": tuning["best_cv_pr_auc_mean"], "pr_auc_std": tuning["best_cv_pr_auc_std"],
                 "source": "RandomizedSearchCV (same folds as the untuned candidates)"}
    fitted, tuned_result = evaluate_candidate(model, imbalance, split, params=tuning["best_params"],
                                              cv_result=cv_result, label=f"{model}[{imbalance}, tuned]")
    tuned_result["stage"] = "tuned"
    tuned_result["mean_fit_time_in_cv_s"] = best_cv["mean_fit_time_s"]
    final = fitted if improved else make_model_pipeline(
        model, make_estimator(model, imbalance, scale_pos_weight=positive_weight(y_train))).fit(X_train, y_train)
    tuning["final_model_size_mb_compressed"] = compressed_size_mb(final)
    return tuning, tuned_result, final


REPRODUCTION_TOLERANCE = 1e-9  # refitted pipeline must reproduce the saved validation PR-AUC


def load_phase_8_results(split: DataSplit, data_sha256: str, tuning_path=None,
                         comparison_path=None) -> tuple[dict, dict, Pipeline]:
    """Resume after Phase 8 (``--from-phase 9``) without rerunning or rewriting it.

    Reads the saved selection and tuning results, checks they were produced by
    the same search on the same data and library versions, refits the final
    configuration on the training split, and checks that the refit reproduces
    the saved validation PR-AUC. Returns (selection, tuning, final pipeline).
    """
    tuning_path = tuning_path or config.TUNING_RESULTS_PATH
    comparison_path = comparison_path or config.MODEL_COMPARISON_PATH
    for path in (tuning_path, comparison_path):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found; run `python -m src.train` without --from-phase first")
    tuning = json.loads(tuning_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    selection = comparison["selection"]
    model, imbalance = selection["selected_model"], selection["selected_imbalance"]

    if comparison["data_sha256"] != data_sha256:
        raise ValueError(f"{comparison_path.name} was produced from different data; rerun the full pipeline")
    if tuning.get("signature") != search_signature(model, imbalance, data_sha256):
        raise ValueError(f"{tuning_path.name} does not match the current search, data or library versions; "
                         "rerun the full pipeline")
    configuration = tuning["final_configuration"]
    if selection.get("final_configuration") != configuration:
        raise ValueError(f"{comparison_path.name} and {tuning_path.name} disagree on the final configuration")

    label = f"{model}[{imbalance}, tuned]" if configuration == "tuned" else selection["selected_candidate"]
    saved = next(c for c in comparison["candidates"] if c["candidate"] == label)
    params = tuning["best_params"] if configuration == "tuned" else {}

    X_train, y_train = split_xy(split.train)
    X_val, y_val = split_xy(split.validation)
    estimator = make_estimator(model, imbalance, scale_pos_weight=positive_weight(y_train)).set_params(**params)
    final = make_model_pipeline(model, estimator).fit(X_train, y_train)

    pr_auc = probability_metrics(y_val, final.predict_proba(X_val)[:, 1])["pr_auc"]
    if abs(pr_auc - saved["validation"]["pr_auc"]) > REPRODUCTION_TOLERANCE:
        raise ValueError(f"refitted {label} gives validation PR-AUC {pr_auc:.10f}, but Phase 8 saved "
                         f"{saved['validation']['pr_auc']:.10f}; rerun the full pipeline")
    logger.info("Resumed from Phase 8: %s refitted on train, validation PR-AUC %.6f reproduced", label, pr_auc)
    return selection, tuning, final


def run_calibration_and_threshold(final_pipeline: Pipeline, split: DataSplit):
    """SPEC §11 and §10.2, on the validation split only.

    1. Compare sigmoid vs isotonic by out-of-fold Brier and apply the sigmoid preference.
    2. Fit the chosen calibrator on the whole validation split (the deployed model).
    3. Choose the cost-minimising threshold on the deployed model's validation PDs.
    """
    X_val, y_val = split_xy(split.validation)
    comparison = compare_calibration_methods(final_pipeline, X_val, y_val)
    method, reason = choose_calibration_method(comparison)
    logger.info("Calibration (out-of-fold Brier): raw %.5f | sigmoid %.5f | isotonic %.5f -> %s (%s)",
                comparison["raw"]["brier"], comparison["sigmoid"]["brier"], comparison["isotonic"]["brier"],
                method, reason)
    calibrated = fit_calibrator(final_pipeline, X_val, y_val, method)

    val_pd = calibrated.predict_proba(X_val)[:, 1]
    threshold, table = choose_threshold(y_val, val_pd)
    at_threshold = metrics_at_threshold(y_val, val_pd, threshold)
    theory = config.COST_FALSE_POSITIVE / (config.COST_FALSE_POSITIVE + config.COST_FALSE_NEGATIVE)
    logger.info("Threshold %.2f (theoretical optimum for calibrated PDs %.3f): precision %.3f, recall %.3f, "
                "F1 %.3f, flagged %.1f%%, cost %.0f", threshold, theory, at_threshold["precision"],
                at_threshold["recall"], at_threshold["f1"], 100 * at_threshold["flagged_rate"], at_threshold["cost"])
    plot_threshold_analysis(table, threshold, config.FIGURES_DIR / "threshold_analysis_validation.png",
                            title_suffix=" (validation split, calibrated PD)")

    calibration_report = {
        "fitted_on": "validation split only (train-fitted pipeline frozen)",
        "comparison": "5-fold out-of-fold calibration within the validation split",
        "isotonic_min_relative_gain": ISOTONIC_MIN_RELATIVE_GAIN,
        "validation_out_of_fold": comparison,
        "chosen_method": method,
        "reason": reason,
    }
    threshold_report = {
        "chosen_on": "validation split, PDs from the deployed (calibrated) model",
        "rule": "minimise cost_fn * FN + cost_fp * FP; ties -> middle threshold",
        "cost_false_negative": config.COST_FALSE_NEGATIVE,
        "cost_false_positive": config.COST_FALSE_POSITIVE,
        "theoretical_optimum_if_perfectly_calibrated": round(theory, 4),
        "chosen_threshold": threshold,
        "validation_at_threshold": at_threshold,
        "table": table[table["threshold"].isin(DISPLAY_THRESHOLDS)].round(4).to_dict(orient="records"),
        "full_grid_cost": table[["threshold", "cost"]].to_dict(orient="records"),
    }
    return calibrated, calibration_report, threshold_report


def training_percentiles(split: DataSplit) -> dict:
    """1st/99th percentiles of each raw feature on the cleaned training split (for app input warnings)."""
    cleaned = CleaningTransformer().fit_transform(split_xy(split.train)[0])
    return {c: {"p01": float(cleaned[c].quantile(0.01)), "p99": float(cleaned[c].quantile(0.99))}
            for c in config.RAW_FEATURES}


def build_metadata(final_pipeline: Pipeline, selection: dict, tuning: dict,
                   calibration_report: dict, threshold_report: dict, split: DataSplit, data_sha256: str) -> dict:
    """Everything the app and later phases need to use the artifact correctly (SPEC §15.1)."""
    estimator = final_pipeline.named_steps["model"]
    return {
        "artifact": config.PIPELINE_PATH.name,
        "pipeline_steps": ["clean", "features", "preprocess", "model", f"calibrator ({calibration_report['chosen_method']})"],
        "raw_input_features": config.RAW_FEATURES,
        "model_input_features": [str(f) for f in final_pipeline[:-1].get_feature_names_out()],
        "target": {"name": config.TARGET, "source_column": config.TARGET_SOURCE,
                   "definition": "90+ days past due within 2 years"},
        "model": {
            "type": selection["selected_model"],
            "class": type(estimator).__name__,
            "imbalance": selection["selected_imbalance"],
            "configuration": tuning["final_configuration"],
            "hyperparameters": {k: v for k, v in estimator.get_params().items()
                                if k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                                         "max_samples", "class_weight", "criterion", "bootstrap", "random_state")},
        },
        "calibration_method": calibration_report["chosen_method"],
        "decision_threshold": threshold_report["chosen_threshold"],
        "cost_false_negative": config.COST_FALSE_NEGATIVE,
        "cost_false_positive": config.COST_FALSE_POSITIVE,
        "risk_bands": {"low_upper": config.RISK_BAND_LOW_UPPER, "medium_upper": config.RISK_BAND_MEDIUM_UPPER},
        "training_percentiles": training_percentiles(split),
        "validation_metrics": {
            "raw": calibration_report["validation_out_of_fold"]["raw"],
            "calibrated_out_of_fold": calibration_report["validation_out_of_fold"][calibration_report["chosen_method"]],
            "at_threshold": threshold_report["validation_at_threshold"],
        },
        "data": {"sha256": data_sha256, "train_rows": len(split.train), "validation_rows": len(split.validation),
                 "train_default_rate": float(split.train[config.TARGET].mean())},
        "random_state": config.RANDOM_STATE,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "shap": importlib.metadata.version("shap"), "pandas": pd.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
    }


def save_artifact(calibrated, metadata: dict) -> None:
    """Write the single model artifact and its metadata (SPEC §15.1)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, config.PIPELINE_PATH, compress=3)
    write_json(config.METADATA_PATH, metadata)
    logger.info("Saved %s (%.1f MB)", config.PIPELINE_PATH.relative_to(config.PROJECT_ROOT),
                config.PIPELINE_PATH.stat().st_size / 1e6)


def run_phase_9(final_pipeline: Pipeline, selection: dict, tuning: dict, split: DataSplit, data_sha256: str):
    """Calibrate, choose the threshold, and write the Phase 9 reports, artifact and metadata.
    Returns the calibrated artifact."""
    calibrated, calibration_report, threshold_report = run_calibration_and_threshold(final_pipeline, split)
    write_json(config.CALIBRATION_RESULTS_PATH, calibration_report)
    write_json(config.THRESHOLD_ANALYSIS_PATH, threshold_report)
    save_artifact(calibrated, build_metadata(final_pipeline, selection, tuning, calibration_report,
                                             threshold_report, split, data_sha256))
    return calibrated


def write_json(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy scalars (e.g. sampled hyperparameters) are converted to plain Python values
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    logger.info("Written %s", path.relative_to(config.PROJECT_ROOT))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train, compare, select and tune models (test split never used).")
    parser.add_argument("--retune", action="store_true",
                        help="run the hyperparameter search even if an identical cached search exists")
    parser.add_argument("--from-phase", type=int, choices=[9], default=None,
                        help="9: reuse the saved Phase 6-8 results (no reruns, reports left unchanged) and run "
                             "only calibration, threshold and artifact saving")
    args = parser.parse_args(argv)
    if args.retune and args.from_phase:
        parser.error("--retune cannot be combined with --from-phase")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    start = time.perf_counter()
    data_sha256 = file_sha256(config.RAW_DATA_FILE)
    split = split_data(load_raw_data())
    logger.info("Train %d rows, validation %d rows (test split not used)", len(split.train), len(split.validation))

    if args.from_phase == 9:
        selection, tuning, final_pipeline = load_phase_8_results(split, data_sha256)
        calibrated = run_phase_9(final_pipeline, selection, tuning, split, data_sha256)
        run_global_explanation(calibrated, split)
        logger.info("Total time %.1fs", time.perf_counter() - start)
        return

    candidates = run_candidates(split)
    selection = select_model(candidates)
    logger.info("Imbalance choice per model: %s", selection["imbalance_choice"])
    logger.info("Selected: %s (%s)", selection["selected_candidate"], selection["reason"])

    tuning, tuned_result, final_pipeline = run_tuning(selection, candidates, split, data_sha256, retune=args.retune)
    candidates.append(tuned_result)
    selection["final_configuration"] = tuning["final_configuration"]
    write_json(config.TUNING_RESULTS_PATH, tuning)

    calibrated = run_phase_9(final_pipeline, selection, tuning, split, data_sha256)
    run_global_explanation(calibrated, split)

    report = {
        "data_sha256": data_sha256,
        "random_state": config.RANDOM_STATE,
        "cv_folds": config.CV_FOLDS,
        "primary_metric": "pr_auc",
        "tie_tolerance": TIE_TOLERANCE,
        "calibrated_brier_method": "sigmoid, 5-fold out-of-fold on the validation split",
        "candidates": candidates,
        "selection": selection,
    }
    write_json(config.MODEL_COMPARISON_PATH, report)
    logger.info("Total time %.1fs", time.perf_counter() - start)


if __name__ == "__main__":
    main()
