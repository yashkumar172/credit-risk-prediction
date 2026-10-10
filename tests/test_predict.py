"""Prediction layer (SPEC §14.2, §15.1, §19.4). Synthetic artifact; one real-artifact test
is skipped when the artifact is absent."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.explain import risk_band
from src.feature_engineering import CleaningTransformer
from src.predict import (
    INPUT_SPECS,
    MAX_AGE,
    MIN_AGE,
    UNRELIABLE,
    cleaned_to_unknown,
    input_warnings,
    load_predictor,
    to_frame,
    validate_inputs,
)
from tests.fixtures.make_synthetic_artifact import write_synthetic_artifact

DEFAULTS = {name: spec.default for name, spec in INPUT_SPECS.items()}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return write_synthetic_artifact(tmp_path_factory.mktemp("artifact"))


@pytest.fixture(scope="module")
def predictor(synthetic):
    pipeline_path, metadata_path, _ = synthetic
    return load_predictor(pipeline_path, metadata_path)


# --- input specs and validation -------------------------------------------------------------


def test_every_raw_feature_has_an_input_spec_and_label():
    assert list(INPUT_SPECS) == config.RAW_FEATURES  # same features, same order
    assert all(name in config.FEATURE_LABELS for name in INPUT_SPECS)
    assert {n for n, s in INPUT_SPECS.items() if s.allow_unknown} == {"MonthlyIncome", "NumberOfDependents"}


def test_default_inputs_are_valid():
    assert validate_inputs(DEFAULTS, config.RAW_FEATURES) == []
    for spec in INPUT_SPECS.values():
        assert spec.min_value <= spec.default <= (spec.max_value or math.inf)


def test_past_due_inputs_cannot_reach_the_special_codes():
    for name in config.PAST_DUE_COLUMNS:
        assert INPUT_SPECS[name].max_value < config.PAST_DUE_CODE_MIN


@pytest.mark.parametrize("change, message", [
    ({"DebtRatio": -0.1}, "cannot be negative"),
    ({"age": MIN_AGE - 1}, "between"),
    ({"age": MAX_AGE + 1}, "between"),
    ({"age": float("nan")}, "Age is required"),
    ({"MonthlyIncome": "abc"}, "must be a number"),
    ({"DebtRatio": float("inf")}, "finite"),
])
def test_invalid_inputs_are_rejected(change, message):
    errors = validate_inputs({**DEFAULTS, **change}, config.RAW_FEATURES)
    assert any(message in e for e in errors)


def test_missing_and_extra_inputs_are_rejected():
    values = {k: v for k, v in DEFAULTS.items() if k != "age"} | {"row_id": 1}
    errors = validate_inputs(values, config.RAW_FEATURES)
    assert any("missing inputs: age" in e for e in errors)
    assert any("unexpected inputs: row_id" in e for e in errors)


def test_unknown_income_and_dependents_are_allowed():
    values = {**DEFAULTS, "MonthlyIncome": float("nan"), "NumberOfDependents": None}
    assert validate_inputs(values, config.RAW_FEATURES) == []
    row = to_frame(values, config.RAW_FEATURES)
    assert list(row.columns) == config.RAW_FEATURES
    assert row["MonthlyIncome"].isna().all() and row["NumberOfDependents"].isna().all()


# --- warnings ---------------------------------------------------------------------------------


PERCENTILES = {name: {"p01": 0.0, "p99": 10_000.0} for name in config.RAW_FEATURES}  # defaults inside


def test_out_of_range_input_warns():
    warnings = input_warnings({**DEFAULTS, "NumberOfOpenCreditLinesAndLoans": 50_000}, PERCENTILES)
    assert any("Open credit lines" in w and UNRELIABLE in w for w in warnings)
    assert input_warnings({**DEFAULTS, "MonthlyIncome": 50}, PERCENTILES) == []


def test_unknown_or_zero_income_explains_the_ignored_debt_ratio():
    for income in (float("nan"), 0):
        warnings = input_warnings({**DEFAULTS, "MonthlyIncome": income}, PERCENTILES)
        assert len(warnings) == 1 and "ignores the debt ratio" in warnings[0]  # no range warning for 0


def test_utilization_above_ten_is_explained_not_range_checked():
    warnings = input_warnings({**DEFAULTS, "RevolvingUtilizationOfUnsecuredLines": 500.0}, PERCENTILES)
    assert len(warnings) == 1 and "treated as unknown" in warnings[0]


def test_cleaned_to_unknown_matches_the_cleaning_step():
    """The app's notes about unknown values must agree with what the pipeline actually does."""
    cases = [("MonthlyIncome", 0), ("MonthlyIncome", 1), ("RevolvingUtilizationOfUnsecuredLines", 10.5),
             ("RevolvingUtilizationOfUnsecuredLines", 10.0), ("NumberOfTimes90DaysLate", 96),
             ("NumberOfTimes90DaysLate", 89), ("age", 17), ("age", 18)]
    for name, value in cases:
        row = to_frame({**DEFAULTS, name: value}, config.RAW_FEATURES)
        cleaned = CleaningTransformer().fit_transform(row)
        assert pd.isna(cleaned[name].iloc[0]) == cleaned_to_unknown(name, float(value)), (name, value)


# --- the predictor (SPEC §19.4) ----------------------------------------------------------------


def test_prediction_is_consistent(predictor):
    result = predictor.predict(DEFAULTS)
    assert 0 <= result.pd <= 1
    assert result.risk_band == risk_band(result.pd)
    assert result.flagged == (result.pd >= predictor.threshold)
    assert result.explanation.pd == result.pd
    assert result.pd == pytest.approx(predictor.artifact.predict_proba(to_frame(DEFAULTS, config.RAW_FEATURES))[0, 1])


def test_flag_follows_the_threshold(synthetic, tmp_path):
    pipeline_path, _, metadata = synthetic
    for threshold, flagged in ((0.0, True), (1.01, False)):
        path = tmp_path / f"metadata_{threshold}.json"
        path.write_text(json.dumps({**metadata, "decision_threshold": threshold}))
        assert load_predictor(pipeline_path, path).predict(DEFAULTS).flagged is flagged


def test_unknown_income_and_dependents_are_predicted(predictor):
    result = predictor.predict({**DEFAULTS, "MonthlyIncome": float("nan"), "NumberOfDependents": float("nan")})
    assert 0 <= result.pd <= 1
    shown = {f.feature: f.display_value for f in result.explanation.factors}
    assert shown["MonthlyIncome"] == "Unknown" and shown["DebtRatio"] == "Unknown"
    assert any("ignores the debt ratio" in w for w in result.warnings)


def test_invalid_inputs_raise(predictor):
    with pytest.raises(ValueError, match="cannot be negative"):
        predictor.predict({**DEFAULTS, "DebtRatio": -1})


def test_version_mismatch_is_reported(synthetic, tmp_path):
    pipeline_path, _, metadata = synthetic
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({**metadata, "versions": {**metadata["versions"], "scikit-learn": "0.1"}}))
    warnings = load_predictor(pipeline_path, path).version_warnings
    assert len(warnings) == 1 and "scikit-learn" in warnings[0] and "0.1" in warnings[0]


def test_missing_artifact_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_predictor(tmp_path / "missing.joblib", tmp_path / "missing.json")


@pytest.mark.skipif(not config.PIPELINE_PATH.exists(), reason="trained artifact not present")
def test_real_artifact_loads_and_predicts():
    predictor = load_predictor()
    assert predictor.raw_features == config.RAW_FEATURES
    assert predictor.version_warnings == []
    result = predictor.predict(DEFAULTS)
    assert 0 <= result.pd <= 1 and result.risk_band in ("LOW", "MEDIUM", "HIGH")
    assert np.isclose(result.explanation.base_value + sum(f.contribution for f in result.explanation.factors),
                      result.explanation.raw_output)
