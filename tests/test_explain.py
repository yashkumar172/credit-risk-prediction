"""SHAP explanation tests (SPEC §13, §19.6). Synthetic data only, except one test that
uses the real artifact and is skipped when it is absent."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src import config
from src.calibration import fit_calibrator
from src.data_loader import DataSplit, load_raw_data, split_data
from src.explain import (
    LOG_ODDS,
    PROBABILITY,
    example_borrower,
    explain_row,
    feature_label,
    format_contribution,
    format_local_explanation,
    format_value,
    global_explanation,
    local_example,
    make_explainer,
    plot_local_explanation,
    risk_band,
    run_global_explanation,
    stratified_sample,
    unwrap_artifact,
)
from src.preprocessing import make_model_pipeline
from src.train import make_estimator, split_xy
from tests.fixtures.make_synthetic_data import write_synthetic_csv


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    path = write_synthetic_csv(tmp_path_factory.mktemp("data") / "cs-training.csv", n_rows=1000)
    s = split_data(load_raw_data(path))
    # Replace the test split with something unusable: any access would raise.
    return DataSplit(train=s.train, validation=s.validation, test=pd.DataFrame({"not": ["a", "dataset"]}))


def fit(model_name, split, **params):
    X, y = split_xy(split.train)
    return make_model_pipeline(model_name, make_estimator(model_name, n_jobs=1).set_params(**params)).fit(X, y)


@pytest.fixture(scope="module")
def rf_artifact(split):
    X_val, y_val = split_xy(split.validation)
    return fit_calibrator(fit("random_forest", split, n_estimators=30), X_val, y_val, "isotonic")


# --- unwrapping ---------------------------------------------------------------------------


def test_unwrap_returns_the_frozen_train_fitted_pipeline(split, rf_artifact):
    pipeline = unwrap_artifact(rf_artifact)
    assert [name for name, _ in pipeline.steps] == ["clean", "features", "preprocess", "model"]
    X_val, _ = split_xy(split.validation)
    raw = rf_artifact.calibrated_classifiers_[0].estimator.predict_proba(X_val)
    np.testing.assert_allclose(pipeline.predict_proba(X_val), raw)
    assert unwrap_artifact(pipeline) is pipeline  # an uncalibrated pipeline is accepted as is


def test_unwrap_rejects_cross_validated_calibrator_ensembles(split):
    X, y = split_xy(split.train)
    ensemble = CalibratedClassifierCV(fit("random_forest", split, n_estimators=10), cv=2).fit(X, y)
    with pytest.raises(ValueError, match="one calibrated classifier"):
        unwrap_artifact(ensemble)
    with pytest.raises(TypeError):
        unwrap_artifact("not a model")


# --- SHAP values add up to the model output, for every supported model ----------------------


@pytest.mark.parametrize("model_name, params, units", [
    ("random_forest", {"n_estimators": 30}, PROBABILITY),
    ("xgboost", {"n_estimators": 30}, LOG_ODDS),
    ("logistic_regression", {}, LOG_ODDS),
])
def test_shap_values_are_additive(split, model_name, params, units):
    pipeline = fit(model_name, split, **params)
    background = split_xy(split.train)[0] if model_name == "logistic_regression" else None
    explainer = make_explainer(pipeline, background=background)
    assert explainer.units == units

    X_val, _ = split_xy(split.validation)
    Z = explainer.transform(X_val)
    assert Z.isna().any(axis=None) == (model_name != "logistic_regression")  # rows with NaN are covered
    values = explainer.shap_values(Z)
    assert values.shape == (len(X_val), len(explainer.feature_names))
    np.testing.assert_allclose(explainer.base_value + values.sum(axis=1), explainer.raw_output(Z), atol=1e-6)
    if units == PROBABILITY:
        np.testing.assert_allclose(explainer.raw_output(Z), pipeline.predict_proba(X_val)[:, 1])
    else:  # log-odds: the sigmoid of the explained output is the model's probability
        np.testing.assert_allclose(1 / (1 + np.exp(-explainer.raw_output(Z))), pipeline.predict_proba(X_val)[:, 1],
                                   rtol=1e-5)


def test_logistic_regression_needs_background(split):
    with pytest.raises(ValueError, match="background"):
        make_explainer(fit("logistic_regression", split))


# --- labels, values and bands ---------------------------------------------------------------


def test_every_model_feature_has_a_label():
    assert set(config.FEATURE_LABELS) == set(config.MODEL_FEATURES)
    assert feature_label("missingindicator_MonthlyIncome") == "Monthly income unknown"
    assert feature_label("age") == "Age"


@pytest.mark.parametrize("name, value, expected", [
    ("MonthlyIncome", np.nan, "Unknown"),
    ("NumberOfTimes90DaysLate", 2.0, "2"),
    ("RevolvingUtilizationOfUnsecuredLines", 0.9712, "0.97"),
    ("MonthlyIncome", 5400.0, "5,400"),
    ("income_per_dependent", 1833.333, "1,833"),
    ("utilization_over_limit", 1.0, "Yes"),
    ("missingindicator_MonthlyIncome", 0.0, "No"),
])
def test_format_value(name, value, expected):
    assert format_value(name, value) == expected


def test_format_contribution_units():
    assert format_contribution(0.0412, PROBABILITY) == "+4.1 pp"
    assert format_contribution(-0.38, LOG_ODDS) == "-0.38 log-odds"
    assert format_contribution(0.0003, PROBABILITY) == "+0.03 pp"  # small effects are not shown as "+0.0"
    assert format_contribution(-0.0009, PROBABILITY) == "-0.09 pp"


@pytest.mark.parametrize("pd_value, band", [
    (0.0, "LOW"), (config.RISK_BAND_LOW_UPPER - 1e-9, "LOW"), (config.RISK_BAND_LOW_UPPER, "MEDIUM"),
    (config.RISK_BAND_MEDIUM_UPPER, "HIGH"), (1.0, "HIGH"),
])
def test_risk_band_boundaries(pd_value, band):
    assert risk_band(pd_value) == band


# --- local explanation (SPEC §13.3) and the end-to-end path (SPEC §19.6) -------------------


def test_local_explanation_end_to_end(split, rf_artifact):
    """synthetic raw input → pipeline → calibrator → PD → band → SHAP factors."""
    explainer = make_explainer(rf_artifact)
    X_val, _ = split_xy(split.validation)
    row = X_val.iloc[[0]][config.RAW_FEATURES].copy()
    row["MonthlyIncome"] = np.nan          # "Unknown" in the app
    row["NumberOfTimes90DaysLate"] = 98    # special code -> cleaned to missing

    local = explain_row(explainer, rf_artifact, row)
    assert local.pd == pytest.approx(rf_artifact.predict_proba(row)[0, 1])
    assert local.risk_band == risk_band(local.pd)
    assert local.base_value + sum(f.contribution for f in local.factors) == pytest.approx(local.raw_output)

    assert len(local.increasing) <= 3 and len(local.decreasing) <= 3
    assert all(f.contribution > 0 for f in local.increasing)
    assert all(f.contribution < 0 for f in local.decreasing)
    up = [f.contribution for f in local.increasing]
    down = [f.contribution for f in local.decreasing]
    assert up == sorted(up, reverse=True) and down == sorted(down)  # strongest first

    shown = {f.feature: f for f in local.factors}
    assert set(shown) == set(config.MODEL_FEATURES)
    assert shown["MonthlyIncome"].display_value == "Unknown"
    assert shown["NumberOfTimes90DaysLate"].display_value == "Unknown"  # value after cleaning, not 98
    assert shown["age"].display_value == format_value("age", row["age"].iloc[0])
    assert shown["age"].label == "Age"


def test_local_explanation_text_uses_only_shap_and_inputs(split, rf_artifact):
    explainer = make_explainer(rf_artifact)
    local = explain_row(explainer, rf_artifact, split_xy(split.validation)[0].iloc[[1]])
    text = format_local_explanation(local)
    assert text.startswith(f"Probability of Default: {100 * local.pd:.1f}%   (Risk band: {local.risk_band})")
    assert "Factors increasing risk:" in text and "Factors reducing risk:" in text
    for f in local.increasing + local.decreasing:
        assert f"{f.label}: {f.display_value}" in text
        assert format_contribution(f.contribution, local.units) in text
    assert "do not add up exactly to the calibrated PD" in text  # SPEC §13.1 note
    for word in ("high income", "low income", "good", "bad"):
        assert word not in text.lower()


def test_explain_row_takes_one_row(split, rf_artifact):
    with pytest.raises(ValueError, match="exactly one row"):
        explain_row(make_explainer(rf_artifact), rf_artifact, split_xy(split.validation)[0].iloc[:2])


def test_local_plot(split, rf_artifact, tmp_path):
    local = explain_row(make_explainer(rf_artifact), rf_artifact, split_xy(split.validation)[0].iloc[[2]])
    fig = plot_local_explanation(local)
    fig.savefig(tmp_path / "local.png")
    assert (tmp_path / "local.png").stat().st_size > 0


# --- the invented example borrower ---------------------------------------------------------


def test_example_borrower_is_a_valid_raw_input():
    row = example_borrower()
    assert list(row.columns) == config.RAW_FEATURES and len(row) == 1
    assert row.notna().all(axis=None) and (row >= 0).all(axis=None)
    assert 18 <= row["age"].iloc[0] <= 110
    assert (row[config.PAST_DUE_COLUMNS] < config.PAST_DUE_CODE_MIN).all(axis=None)  # real counts, not codes


def test_example_borrower_is_not_taken_from_the_data(split):
    """Defence in depth on synthetic data; the real file was checked when the example was made."""
    X = split_xy(split.validation)[0][config.RAW_FEATURES]
    assert not (X == example_borrower().iloc[0]).all(axis=1).any()


def test_local_example_entry(rf_artifact):
    entry, fig = local_example(make_explainer(rf_artifact), rf_artifact)
    local = explain_row(make_explainer(rf_artifact), rf_artifact, example_borrower())
    assert entry["text"] == format_local_explanation(local)
    assert entry["inputs"] == config.SHAP_EXAMPLE_BORROWER
    assert len(fig.axes) == 1


# --- global explanation (SPEC §13.2) -----------------------------------------------------------


def test_stratified_sample_is_fixed_and_keeps_the_default_rate(split):
    X, y = split_xy(split.train)
    X1, y1 = stratified_sample(X, y, n=200)
    X2, _ = stratified_sample(X, y, n=200)
    assert len(X1) == 200
    assert X1.index.equals(X2.index)  # fixed seed
    assert y1.mean() == pytest.approx(y.mean(), abs=0.01)
    X_all, _ = stratified_sample(X, y, n=10 * len(X))
    assert len(X_all) == len(X)


def test_global_report(split, rf_artifact):
    explainer = make_explainer(rf_artifact)
    X_val, y_val = split_xy(split.validation)
    report, values, Z = global_explanation(explainer, X_val, y_val, n=100)
    assert report["sample"]["rows"] == 100 == len(Z) == len(values)
    assert report["units"] == PROBABILITY
    assert report["max_additivity_error"] < 1e-6
    ranks = [f["mean_abs_shap"] for f in report["features"]]
    assert ranks == sorted(ranks, reverse=True)
    assert {f["feature"] for f in report["features"]} == set(config.MODEL_FEATURES)
    assert report["top_features"] == [f["label"] for f in report["features"][:5]]


def test_run_global_explanation_writes_reports(monkeypatch, tmp_path, split, rf_artifact):
    for name, path in (("SHAP_GLOBAL_PATH", tmp_path / "shap_global.json"),
                       ("SHAP_BEESWARM_FIGURE", tmp_path / "beeswarm.png"),
                       ("SHAP_BAR_FIGURE", tmp_path / "bar.png"),
                       ("SHAP_LOCAL_EXAMPLE_FIGURE", tmp_path / "local.png")):
        monkeypatch.setattr(config, name, path)
    report = run_global_explanation(rf_artifact, split)  # the test split is unusable: never read
    for name in ("beeswarm.png", "bar.png", "local.png"):
        assert (tmp_path / name).stat().st_size > 0
    saved = json.loads((tmp_path / "shap_global.json").read_text())
    assert saved["sample"]["source"] == "validation split"
    example = saved["local_example"]
    assert "row_id" not in example and example["source"].startswith("synthetic")  # no dataset row is written
    assert example["inputs"] == config.SHAP_EXAMPLE_BORROWER
    assert example["text"].startswith("Probability of Default:")
    assert saved["top_features"] == report["top_features"]


# --- the real artifact (skipped when absent, SPEC §19.1) ------------------------------------


@pytest.mark.skipif(not config.PIPELINE_PATH.exists(), reason="trained artifact not present")
def test_real_artifact_explains_synthetic_rows(split):
    artifact = joblib.load(config.PIPELINE_PATH)
    explainer = make_explainer(artifact)
    X = split_xy(split.validation)[0].iloc[:20]
    Z = explainer.transform(X)
    np.testing.assert_allclose(explainer.base_value + explainer.shap_values(Z).sum(axis=1),
                               explainer.raw_output(Z), atol=1e-6)
    local = explain_row(explainer, artifact, X.iloc[[0]])
    assert 0 <= local.pd <= 1 and local.risk_band in ("LOW", "MEDIUM", "HIGH")
