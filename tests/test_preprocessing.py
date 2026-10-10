"""Model-specific preprocessing tests (SPEC §8). No Kaggle data needed."""

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src import config
from src.data_loader import load_raw_data
from src.feature_engineering import make_feature_pipeline
from src.preprocessing import QuantileClipper, make_logistic_preprocessor, make_model_pipeline, make_preprocessor

EXPECTED_LR_COLUMNS = [
    *config.SKEWED_FEATURES,
    *config.UNSKEWED_FEATURES,
    *config.BINARY_FEATURES,
    *[f"missingindicator_{c}" for c in config.MISSING_INDICATOR_FEATURES],
]


@pytest.fixture
def raw(synthetic_csv):
    df = load_raw_data(synthetic_csv)
    return df.drop(columns=config.TARGET), df[config.TARGET]


@pytest.fixture
def engineered(raw):
    return make_feature_pipeline().fit_transform(raw[0])


# --- QuantileClipper ------------------------------------------------------------------------


def test_clipper_uses_bounds_learned_in_fit_only():
    train = np.arange(101, dtype=float).reshape(-1, 1)  # 0..100
    clipper = QuantileClipper(lower=0.01, upper=0.99).fit(train)
    out = clipper.transform(np.array([[-50.0], [50.0], [500.0]]))
    np.testing.assert_allclose(out.ravel(), [1.0, 50.0, 99.0])


def test_clipper_ignores_and_preserves_nan():
    train = np.array([[0.0], [np.nan], [10.0], [20.0]])
    clipper = QuantileClipper(lower=0.0, upper=1.0).fit(train)
    np.testing.assert_allclose(clipper.lower_bounds_, [0.0])
    np.testing.assert_allclose(clipper.upper_bounds_, [20.0])
    assert np.isnan(clipper.transform(np.array([[np.nan]]))[0, 0])


# --- Logistic Regression preprocessor -------------------------------------------------------


def test_logistic_output_is_complete_and_named(engineered):
    pre = make_logistic_preprocessor()
    out = pre.fit_transform(engineered)
    assert list(pre.get_feature_names_out()) == EXPECTED_LR_COLUMNS
    assert out.shape == (len(engineered), len(EXPECTED_LR_COLUMNS))
    assert np.isfinite(out).all()


def test_logistic_scaled_columns_are_standardized_on_fit_data(engineered):
    out = make_logistic_preprocessor().fit_transform(engineered)
    n_scaled = len(config.SKEWED_FEATURES) + len(config.UNSKEWED_FEATURES)
    scaled = out[:, :n_scaled]
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-9)
    stds = scaled.std(axis=0)
    assert np.all((np.isclose(stds, 1)) | (np.isclose(stds, 0)))  # constant columns scale to 0


def test_missing_indicators_match_missing_pattern(engineered):
    pre = make_logistic_preprocessor()
    out = pre.fit_transform(engineered)
    names = list(pre.get_feature_names_out())
    for col in config.MISSING_INDICATOR_FEATURES:
        indicator = out[:, names.index(f"missingindicator_{col}")]
        np.testing.assert_array_equal(indicator, engineered[col].isna().to_numpy().astype(float))
    # the 96/98-coded rows (6 in the fixture) are flagged
    assert out[:, names.index("missingindicator_NumberOfTimes90DaysLate")].sum() == 6


def test_rare_missing_indicator_is_not_clipped_away(engineered):
    """Regression test: a 0/1 column set in <1% of rows must keep its 1s."""
    data = engineered.copy()
    data["NumberOfTimes90DaysLate"] = data["NumberOfTimes90DaysLate"].fillna(0)
    data.loc[data.index[0], "NumberOfTimes90DaysLate"] = np.nan  # 1 of 300 rows
    pre = make_logistic_preprocessor()
    out = pre.fit_transform(data)
    col = list(pre.get_feature_names_out()).index("missingindicator_NumberOfTimes90DaysLate")
    assert out[:, col].sum() == 1


def test_missing_indicator_reported_even_without_missing_values(engineered):
    complete = engineered.fillna(0)
    pre = make_logistic_preprocessor()
    assert pre.fit_transform(complete).shape[1] == len(EXPECTED_LR_COLUMNS)


def test_logistic_statistics_come_from_fit_data_only(engineered):
    first, second = engineered.iloc[:150], engineered.iloc[150:]
    pre = make_logistic_preprocessor().fit(first)
    age_imputer = pre.named_transformers_["unskewed"].named_steps["impute"]
    assert age_imputer.statistics_[0] == first["age"].median()
    pre.transform(second)  # transforming new data does not refit
    assert pre.named_transformers_["unskewed"].named_steps["impute"].statistics_[0] == first["age"].median()


# --- per-model selection and full pipelines -------------------------------------------------


@pytest.mark.parametrize("model_name", ["random_forest", "xgboost"])
def test_tree_models_get_no_preprocessing(model_name):
    assert make_preprocessor(model_name) == "passthrough"


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_preprocessor("svm")


ESTIMATORS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=10, random_state=0),
    "xgboost": XGBClassifier(n_estimators=10, tree_method="hist", device="cpu", random_state=0),
}


@pytest.mark.parametrize("model_name", config.MODEL_NAMES)
def test_full_pipeline_fits_on_raw_data_with_missing_values(raw, model_name, tmp_path):
    X, y = raw
    assert X.isna().any(axis=None)  # the fixture contains missing values
    pipe = make_model_pipeline(model_name, ESTIMATORS[model_name]).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))

    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(X)[:, 1], proba)


@pytest.mark.parametrize("model_name, expected", [
    ("logistic_regression", EXPECTED_LR_COLUMNS),
    ("random_forest", config.MODEL_FEATURES),
    ("xgboost", config.MODEL_FEATURES),
])
def test_model_input_feature_names(raw, model_name, expected):
    X, y = raw
    pipe = make_model_pipeline(model_name, ESTIMATORS[model_name]).fit(X, y)
    names = list(pipe[:-1].get_feature_names_out())
    assert names == expected
    assert config.TARGET not in names
