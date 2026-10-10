"""Calibration, threshold and artifact tests (SPEC §10.2, §11, §15.1). Synthetic data only."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config, train
from src.calibration import choose_calibration_method, fit_calibrator
from src.data_loader import DataSplit, load_raw_data, split_data
from src.evaluate import choose_threshold, metrics_at_threshold, plot_threshold_analysis, threshold_table
from src.preprocessing import make_model_pipeline
from src.train import (
    build_metadata,
    make_estimator,
    run_calibration_and_threshold,
    run_candidates,
    save_artifact,
    select_model,
    split_xy,
)
from tests.fixtures.make_synthetic_data import write_synthetic_csv

# --- calibration rule -----------------------------------------------------------------------


def comparison(sigmoid, isotonic):
    return {"raw": {"brier": 0.05}, "sigmoid": {"brier": sigmoid}, "isotonic": {"brier": isotonic}}


@pytest.mark.parametrize("sigmoid, isotonic, expected", [
    (0.0500, 0.0490, "isotonic"),   # 2% better -> clearly better
    (0.0500, 0.0495, "isotonic"),   # exactly 1% better -> meets the rule
    (0.0500, 0.0497, "sigmoid"),    # 0.6% better -> not clearly better
    (0.0500, 0.0510, "sigmoid"),    # worse
])
def test_calibration_method_rule(sigmoid, isotonic, expected):
    method, reason = choose_calibration_method(comparison(sigmoid, isotonic))
    assert method == expected
    assert reason


# --- threshold metrics ----------------------------------------------------------------------


def test_metrics_at_threshold_known_values():
    y = [0, 0, 0, 1, 1]
    proba = [0.1, 0.3, 0.6, 0.4, 0.9]
    m = metrics_at_threshold(y, proba, 0.35, cost_fn=5, cost_fp=1)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert m["flagged_rate"] == pytest.approx(0.6)
    assert m["cost"] == 1


def test_no_flagged_rows_gives_zero_precision_not_error():
    m = metrics_at_threshold([0, 1], [0.1, 0.2], 0.99)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["cost"] == config.COST_FALSE_NEGATIVE


def test_choose_threshold_minimises_cost_and_takes_middle_of_ties():
    y = np.array([0] * 8 + [1] * 2)
    proba = np.array([0.05] * 8 + [0.5] * 2)
    # any threshold in (0.05, 0.5] flags exactly the two defaulters: cost 0
    threshold, table = choose_threshold(y, proba, thresholds=np.round(np.arange(0.1, 0.6, 0.1), 2), cost_fn=5, cost_fp=1)
    zero_cost = table.loc[table["cost"] == 0, "threshold"].tolist()
    assert zero_cost == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert threshold == 0.3


def test_cost_weights_move_the_threshold():
    rng = np.random.default_rng(0)
    proba = rng.random(2000)
    y = (rng.random(2000) < proba).astype(int)  # perfectly calibrated
    t_high_fn, _ = choose_threshold(y, proba, cost_fn=5, cost_fp=1)
    t_equal, _ = choose_threshold(y, proba, cost_fn=1, cost_fp=1)
    assert t_high_fn < t_equal  # costlier misses -> flag more borrowers
    assert t_high_fn == pytest.approx(1 / 6, abs=0.08)  # near theory for calibrated PDs
    assert t_equal == pytest.approx(0.5, abs=0.08)


def test_threshold_plot_is_written(tmp_path):
    y = [0, 0, 1, 1] * 25
    proba = np.linspace(0.01, 0.99, 100)
    table = threshold_table(y, proba)
    path = tmp_path / "fig.png"
    plot_threshold_analysis(table, 0.5, path)
    assert path.stat().st_size > 0


# --- calibrator and full stage on synthetic data --------------------------------------------


@pytest.fixture
def synthetic_split(tmp_path):
    split = split_data(load_raw_data(write_synthetic_csv(tmp_path / "cs-training.csv", n_rows=1000)))
    # Replace the test split with something unusable: any access would raise.
    return DataSplit(train=split.train, validation=split.validation, test=pd.DataFrame({"not": ["a", "dataset"]}))


@pytest.fixture
def fitted_rf(synthetic_split):
    X, y = split_xy(synthetic_split.train)
    return make_model_pipeline("random_forest", make_estimator("random_forest").set_params(n_estimators=30)).fit(X, y)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_wraps_frozen_pipeline_and_is_monotone(synthetic_split, fitted_rf, method):
    X_val, y_val = split_xy(synthetic_split.validation)
    raw = fitted_rf.predict_proba(X_val)[:, 1]
    calibrated = fit_calibrator(fitted_rf, X_val, y_val, method)
    pd_cal = calibrated.predict_proba(X_val)[:, 1]
    np.testing.assert_allclose(fitted_rf.predict_proba(X_val)[:, 1], raw)  # pipeline not refitted
    assert np.all((pd_cal >= 0) & (pd_cal <= 1))
    order = np.argsort(raw)
    assert np.all(np.diff(pd_cal[order]) >= -1e-12)  # calibration preserves the ranking


def test_full_stage_and_artifact(monkeypatch, tmp_path, synthetic_split, fitted_rf):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    calibrated, cal_report, thr_report = run_calibration_and_threshold(fitted_rf, synthetic_split)

    assert cal_report["chosen_method"] in ("sigmoid", "isotonic")
    assert set(cal_report["validation_out_of_fold"]) == {"raw", "sigmoid", "isotonic"}
    assert 0 < thr_report["chosen_threshold"] < 1
    assert thr_report["validation_at_threshold"]["threshold"] == thr_report["chosen_threshold"]
    assert [r["threshold"] for r in thr_report["table"]] == pytest.approx(list(np.arange(0.05, 0.951, 0.05)))
    assert (tmp_path / "figures" / "threshold_analysis_validation.png").exists()

    candidates = run_candidates(synthetic_split, models=("random_forest",))
    selection = select_model(candidates)
    tuning = {"final_configuration": "untuned"}
    metadata = build_metadata(fitted_rf, selection, tuning, cal_report, thr_report, synthetic_split, "hash")
    assert metadata["raw_input_features"] == config.RAW_FEATURES
    assert metadata["model_input_features"] == config.MODEL_FEATURES
    assert metadata["decision_threshold"] == thr_report["chosen_threshold"]
    assert set(metadata["training_percentiles"]) == set(config.RAW_FEATURES)
    assert all(v["p01"] <= v["p99"] for v in metadata["training_percentiles"].values())
    assert metadata["versions"]["scikit-learn"]

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "PIPELINE_PATH", tmp_path / "models" / "pipeline.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "models" / "metadata.json")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    save_artifact(calibrated, metadata)

    loaded = joblib.load(config.PIPELINE_PATH)
    X_val, _ = split_xy(synthetic_split.validation)
    np.testing.assert_allclose(loaded.predict_proba(X_val)[:, 1], calibrated.predict_proba(X_val)[:, 1])
    assert json.loads(config.METADATA_PATH.read_text())["decision_threshold"] == thr_report["chosen_threshold"]
    # the artifact takes raw input: raw columns in, PD out
    raw_row = X_val.iloc[[0]][config.RAW_FEATURES]
    assert 0 <= loaded.predict_proba(raw_row)[0, 1] <= 1
