"""Cleaning and feature-engineering tests (SPEC §19.3). No Kaggle data needed."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src import config
from src.data_loader import load_raw_data
from src.feature_engineering import CleaningTransformer, FeatureEngineer, make_feature_pipeline

TYPICAL = {
    "RevolvingUtilizationOfUnsecuredLines": 0.5,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "DebtRatio": 0.3,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 2,
    "NumberOfDependents": 1,
}


def rows(*overrides: dict) -> pd.DataFrame:
    """One row per override dict, each starting from a typical borrower."""
    return pd.DataFrame([{**TYPICAL, **o} for o in overrides])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return CleaningTransformer().fit_transform(df)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    return make_feature_pipeline().fit_transform(df)


# --- cleaning rules -------------------------------------------------------------------------


def test_typical_row_unchanged_by_cleaning():
    out = clean(rows({}))
    expected = rows({}).astype(float)
    pd.testing.assert_frame_equal(out, expected)


@pytest.mark.parametrize("value, becomes_nan", [(96, True), (98, True), (90, True), (89, False), (17, False), (0, False)])
@pytest.mark.parametrize("column", config.PAST_DUE_COLUMNS)
def test_past_due_codes_become_missing(column, value, becomes_nan):
    out = clean(rows({column: value}))
    assert np.isnan(out.at[0, column]) == becomes_nan


@pytest.mark.parametrize("age, becomes_nan", [(0, True), (17, True), (18, False), (109, False)])
def test_invalid_age_becomes_missing(age, becomes_nan):
    assert np.isnan(clean(rows({"age": age})).at[0, "age"]) == becomes_nan


def test_zero_income_becomes_missing_and_drops_debt_ratio():
    out = clean(rows({"MonthlyIncome": 0, "DebtRatio": 924.0}))
    assert np.isnan(out.at[0, "MonthlyIncome"])
    assert np.isnan(out.at[0, "DebtRatio"])


def test_missing_income_drops_debt_ratio():
    out = clean(rows({"MonthlyIncome": np.nan, "DebtRatio": 1160.0}))
    assert np.isnan(out.at[0, "DebtRatio"])


def test_debt_ratio_kept_when_income_present():
    assert clean(rows({"DebtRatio": 4.5})).at[0, "DebtRatio"] == 4.5


@pytest.mark.parametrize("value, becomes_nan", [(10.0, False), (10.01, True), (50708.0, True), (1.5, False), (0.9999999, False)])
def test_implausible_utilization_becomes_missing(value, becomes_nan):
    out = clean(rows({"RevolvingUtilizationOfUnsecuredLines": value}))
    assert np.isnan(out.at[0, "RevolvingUtilizationOfUnsecuredLines"]) == becomes_nan


def test_infinite_values_become_missing():
    out = clean(rows({"DebtRatio": np.inf, "RevolvingUtilizationOfUnsecuredLines": -np.inf}))
    assert out[["DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]].isna().all(axis=None)


def test_cleaning_does_not_modify_input():
    df = rows({"age": 0, "MonthlyIncome": 0, "NumberOfTimes90DaysLate": 98})
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


# --- engineered features --------------------------------------------------------------------


def test_total_past_due_is_sum_of_counts():
    out = engineer(rows({}))
    assert out.at[0, "total_past_due"] == 1 + 0 + 2


def test_total_past_due_missing_for_coded_rows():
    out = engineer(rows({c: 98 for c in config.PAST_DUE_COLUMNS}))
    assert np.isnan(out.at[0, "total_past_due"])


@pytest.mark.parametrize("dependents, expected", [(0, 5000.0), (1, 2500.0), (4, 1000.0)])
def test_income_per_dependent(dependents, expected):
    assert engineer(rows({"NumberOfDependents": dependents})).at[0, "income_per_dependent"] == expected


@pytest.mark.parametrize("override", [{"MonthlyIncome": np.nan}, {"MonthlyIncome": 0}, {"NumberOfDependents": np.nan}])
def test_income_per_dependent_missing_when_inputs_missing(override):
    assert np.isnan(engineer(rows(override)).at[0, "income_per_dependent"])


@pytest.mark.parametrize("value, expected", [(0.5, 0.0), (1.0, 0.0), (0.9999999, 0.0), (1.01, 1.0), (10.0, 1.0), (11.0, np.nan)])
def test_utilization_over_limit(value, expected):
    result = engineer(rows({"RevolvingUtilizationOfUnsecuredLines": value})).at[0, "utilization_over_limit"]
    assert result == expected or (np.isnan(expected) and np.isnan(result))


# --- column handling, leakage and robustness ------------------------------------------------


def test_output_columns_and_feature_names():
    pipe = make_feature_pipeline()
    out = pipe.fit_transform(rows({}))
    assert list(out.columns) == config.MODEL_FEATURES
    assert list(pipe.get_feature_names_out()) == config.MODEL_FEATURES


def test_target_and_id_never_reach_features(synthetic_csv):
    df = load_raw_data(synthetic_csv).reset_index()  # includes row_id and default columns
    out = engineer(df)
    assert config.TARGET not in out.columns
    assert config.TARGET_SOURCE not in out.columns
    assert config.ID_INDEX_NAME not in out.columns


def test_column_order_of_input_does_not_matter():
    df = rows({}, {"age": 30})
    shuffled = df[df.columns[::-1]]
    pd.testing.assert_frame_equal(engineer(df), engineer(shuffled))


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="DebtRatio"):
        engineer(rows({}).drop(columns=["DebtRatio"]))


def test_non_dataframe_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        CleaningTransformer().fit(rows({}).to_numpy())


@pytest.mark.parametrize("transformer", [CleaningTransformer(), FeatureEngineer()])
def test_transform_before_fit_raises(transformer):
    with pytest.raises(NotFittedError):
        transformer.transform(rows({}))


def test_synthetic_data_has_no_infinite_values_and_keeps_rows(synthetic_csv):
    df = load_raw_data(synthetic_csv)
    out = engineer(df)
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert not np.isinf(out.to_numpy()).any()
    assert out["total_past_due"].isna().sum() == 6  # 5 rows of 98 + 1 row of 96 in the fixture


def test_pipeline_is_stateless_cloneable_and_persistable(tmp_path, synthetic_csv):
    df = load_raw_data(synthetic_csv)
    pipe = make_feature_pipeline().fit(df.iloc[:50])  # fit on a subset ...
    expected = make_feature_pipeline().fit(df).transform(df)
    pd.testing.assert_frame_equal(pipe.transform(df), expected)  # ... same output: nothing learned

    clone(pipe).fit(df)
    path = tmp_path / "pipe.joblib"
    joblib.dump(pipe, path)
    pd.testing.assert_frame_equal(joblib.load(path).transform(df), expected)
