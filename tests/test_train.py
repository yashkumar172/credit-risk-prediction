"""Evaluation-metric, calibration-helper, training-stage and selection tests. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import config
from src.calibration import calibrated_brier_cv, out_of_fold_calibrated_proba
from src.data_loader import DataSplit, load_raw_data, split_data
from src.evaluate import probability_metrics
from src.preprocessing import make_model_pipeline
from src.train import (
    IMBALANCE_STRATEGIES,
    choose_imbalance,
    cross_validate_pipeline,
    make_estimator,
    run_candidates,
    select_model,
    split_xy,
)
from tests.fixtures.make_synthetic_data import write_synthetic_csv

# --- metrics --------------------------------------------------------------------------------


def test_probability_metrics_known_values():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.35, 0.8]
    m = probability_metrics(y, proba)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)  # AP = mean precision at each positive
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert m["mean_predicted"] == pytest.approx(0.4125)
    assert m["observed_rate"] == pytest.approx(0.5)


def test_perfect_ranking_scores_one():
    m = probability_metrics([0, 0, 1], [0.2, 0.3, 0.9])
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)


# --- estimators -----------------------------------------------------------------------------


@pytest.mark.parametrize("model, imbalance, cls, check", [
    ("logistic_regression", "none", LogisticRegression, lambda e: e.class_weight is None),
    ("logistic_regression", "class_weight", LogisticRegression, lambda e: e.class_weight == "balanced"),
    ("random_forest", "none", RandomForestClassifier, lambda e: e.class_weight is None),
    ("random_forest", "class_weight", RandomForestClassifier, lambda e: e.class_weight == "balanced"),
    ("xgboost", "none", XGBClassifier, lambda e: e.get_params()["scale_pos_weight"] == 1.0),
    ("xgboost", "scale_pos_weight", XGBClassifier, lambda e: e.get_params()["scale_pos_weight"] == 14.0),
])
def test_make_estimator(model, imbalance, cls, check):
    est = make_estimator(model, imbalance, scale_pos_weight=14.0)
    assert isinstance(est, cls)
    assert check(est)


def test_xgboost_is_cpu_only():
    params = make_estimator("xgboost").get_params()
    assert params["device"] == "cpu"
    assert params["tree_method"] == "hist"


@pytest.mark.parametrize("model, imbalance", [("svm", "none"), ("xgboost", "class_weight"), ("random_forest", "smote")])
def test_unknown_estimator_rejected(model, imbalance):
    with pytest.raises(ValueError):
        make_estimator(model, imbalance)


# --- CV, calibration helper and candidate runs on synthetic data ----------------------------


@pytest.fixture
def synthetic_split(tmp_path):
    # 1,000 rows so every CV / calibration fold contains several defaults
    return split_data(load_raw_data(write_synthetic_csv(tmp_path / "cs-training.csv", n_rows=1000)))


def test_cross_validation_result_structure(synthetic_split):
    X, y = split_xy(synthetic_split.train)
    result = cross_validate_pipeline(make_model_pipeline("logistic_regression", make_estimator("logistic_regression")), X, y)
    assert len(result["pr_auc_folds"]) == config.CV_FOLDS
    for metric in ("pr_auc", "roc_auc", "brier"):
        assert 0 <= result[f"{metric}_mean"] <= 1
        assert result[f"{metric}_std"] >= 0


def test_out_of_fold_calibration_does_not_refit_the_model(synthetic_split):
    X_tr, y_tr = split_xy(synthetic_split.train)
    X_va, y_va = split_xy(synthetic_split.validation)
    pipe = make_model_pipeline("xgboost", make_estimator("xgboost")).fit(X_tr, y_tr)
    before = pipe.predict_proba(X_va)[:, 1]
    proba = out_of_fold_calibrated_proba(pipe, X_va, y_va)
    assert proba.shape == (len(X_va),)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(pipe.predict_proba(X_va)[:, 1], before)  # frozen: unchanged
    assert 0 <= calibrated_brier_cv(pipe, X_va, y_va) <= 1


def test_candidates_never_touch_test_split(synthetic_split):
    # Replace the test split with something unusable: any access would raise.
    poisoned = DataSplit(train=synthetic_split.train, validation=synthetic_split.validation,
                         test=pd.DataFrame({"not": ["a", "dataset"]}))
    results = run_candidates(poisoned)
    expected = [(m, i) for m in config.MODEL_NAMES for i in IMBALANCE_STRATEGIES[m]]
    assert [(r["model"], r["imbalance"]) for r in results] == expected
    for r in results:
        assert set(r) >= {"candidate", "cv", "validation", "validation_brier_calibrated", "fit_time_s"}
        assert r["validation"]["observed_rate"] == pytest.approx(synthetic_split.validation[config.TARGET].mean())
    lr = next(r for r in results if r["model"] == "logistic_regression")
    assert len(lr["coefficients"]) == 17  # LR input columns (Phase 5)
    spw = next(r for r in results if r["imbalance"] == "scale_pos_weight")
    y = synthetic_split.train[config.TARGET]
    assert spw["scale_pos_weight"] == pytest.approx((y == 0).sum() / (y == 1).sum(), abs=1e-4)


def test_class_weighting_inflates_probabilities(synthetic_split):
    results = {r["imbalance"]: r for r in run_candidates(synthetic_split, models=("logistic_regression",))}
    assert results["class_weight"]["validation"]["mean_predicted"] > results["none"]["validation"]["mean_predicted"]


# --- selection rules (SPEC §5.3, §12) --------------------------------------------------------


def cand(model, imbalance, cv_pr, val_pr, brier_cal, fit=1.0):
    return {"candidate": f"{model}[{imbalance}]", "model": model, "imbalance": imbalance,
            "cv": {"pr_auc_mean": cv_pr}, "validation": {"pr_auc": val_pr},
            "validation_brier_calibrated": brier_cal, "fit_time_s": fit}


def test_imbalance_tie_prefers_no_reweighting():
    chosen = choose_imbalance([cand("xgboost", "none", 0.380, 0.4, 0.05),
                               cand("xgboost", "scale_pos_weight", 0.384, 0.4, 0.05)])
    assert chosen["xgboost"]["imbalance"] == "none"


def test_imbalance_clear_winner_is_kept():
    chosen = choose_imbalance([cand("xgboost", "none", 0.380, 0.4, 0.05),
                               cand("xgboost", "scale_pos_weight", 0.390, 0.4, 0.05)])
    assert chosen["xgboost"]["imbalance"] == "scale_pos_weight"


def test_selection_clear_winner():
    sel = select_model([cand("logistic_regression", "none", 0.38, 0.400, 0.049),
                        cand("xgboost", "none", 0.39, 0.410, 0.051)])
    assert sel["selected_candidate"] == "xgboost[none]"
    assert sel["tied_with_best"] == ["xgboost[none]"]


def test_selection_tie_broken_by_calibrated_brier():
    sel = select_model([cand("logistic_regression", "none", 0.38, 0.400, 0.0500),
                        cand("xgboost", "none", 0.39, 0.404, 0.0510),
                        cand("random_forest", "none", 0.39, 0.403, 0.0490)])
    assert sel["selected_candidate"] == "random_forest[none]"
    assert len(sel["tied_with_best"]) == 3


def test_selection_equal_brier_prefers_simpler_model():
    sel = select_model([cand("xgboost", "none", 0.39, 0.404, 0.05),
                        cand("logistic_regression", "none", 0.38, 0.401, 0.05)])
    assert sel["selected_model"] == "logistic_regression"


def test_selection_boundary_just_outside_tolerance_is_not_a_tie():
    sel = select_model([cand("random_forest", "none", 0.39, 0.4054, 0.0504),
                        cand("xgboost", "none", 0.39, 0.3994, 0.0509),
                        cand("logistic_regression", "none", 0.38, 0.3994, 0.0495)])
    assert sel["selected_candidate"] == "random_forest[none]"
    assert sel["tied_with_best"] == ["random_forest[none]"]
