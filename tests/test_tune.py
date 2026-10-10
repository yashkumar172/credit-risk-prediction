"""Hyperparameter-tuning tests (SPEC §9). Synthetic data only; small searches for speed."""

import json

import pandas as pd
import pytest

from src import config, train
from src.data_loader import DataSplit, load_raw_data, split_data
from src.train import make_estimator, run_candidates, run_tuning, select_model, split_xy
from src.tune import SEARCH_SPACES, tune_model
from tests.fixtures.make_synthetic_data import write_synthetic_csv


@pytest.fixture
def synthetic_split(tmp_path):
    return split_data(load_raw_data(write_synthetic_csv(tmp_path / "cs-training.csv", n_rows=1000)))


@pytest.mark.parametrize("model", ["random_forest", "xgboost"])
def test_tune_model_result(synthetic_split, model):
    X, y = split_xy(synthetic_split.train)
    est = make_estimator(model, n_jobs=1).set_params(n_estimators=20)
    result = tune_model(model, est, X, y, n_iter=3, n_folds=3)

    assert len(result["trials"]) == 3
    assert set(result["best_params"]) == {k.removeprefix("model__") for k in SEARCH_SPACES[model]}
    scores = [t["cv_pr_auc_mean"] for t in result["trials"]]
    assert scores == sorted(scores, reverse=True)  # best first
    assert result["best_cv_pr_auc_mean"] == pytest.approx(scores[0], abs=1e-5)
    assert result["tuning_time_s"] >= 0
    json.dumps(result, default=lambda o: o.item() if hasattr(o, "item") else str(o))  # serializable


def test_random_forest_space_values_are_respected(synthetic_split):
    X, y = split_xy(synthetic_split.train)
    result = tune_model("random_forest", make_estimator("random_forest", n_jobs=1), X, y, n_iter=4, n_folds=3)
    space = SEARCH_SPACES["random_forest"]
    for trial in result["trials"]:
        for name, value in trial["params"].items():
            assert value in space[f"model__{name}"]


def test_unknown_model_rejected(synthetic_split):
    X, y = split_xy(synthetic_split.train)
    with pytest.raises(ValueError, match="No search space"):
        tune_model("logistic_regression", make_estimator("logistic_regression"), X, y)


def fake_tuning(best_cv):
    def _tune(model_name, estimator, X, y, **kwargs):
        params = {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": None}
        return {"best_params": params, "best_cv_pr_auc_mean": best_cv, "best_cv_pr_auc_std": 0.01,
                "tuning_time_s": 0.0, "trials": [{"params": params, "mean_fit_time_s": 0.1}]}
    return _tune


@pytest.fixture
def rf_selection(synthetic_split):
    # Replace the test split with something unusable: any access would raise.
    poisoned = DataSplit(train=synthetic_split.train, validation=synthetic_split.validation,
                         test=pd.DataFrame({"not": ["a", "dataset"]}))
    candidates = run_candidates(poisoned, models=("random_forest",))
    return poisoned, candidates, select_model(candidates)


@pytest.mark.parametrize("best_cv, expected", [(0.99, "tuned"), (0.0, "untuned")])
def test_run_tuning_keeps_tuned_only_if_cv_improves(monkeypatch, tmp_path, rf_selection, best_cv, expected):
    monkeypatch.setattr(train, "tune_model", fake_tuning(best_cv))
    split, candidates, selection = rf_selection
    tuning, tuned, final = run_tuning(selection, candidates, split, "hash", cache_path=tmp_path / "tuning.json")

    assert tuning["final_configuration"] == expected
    assert tuning["reused_cached_search"] is False
    assert tuned["candidate"] == f"random_forest[{selection['selected_imbalance']}, tuned]"
    assert tuned["params"]["max_depth"] == 5
    assert tuned["cv"]["pr_auc_mean"] == best_cv  # CV reused from the search, not recomputed
    assert tuning["final_model_size_mb_compressed"] >= 0
    # the returned pipeline is the configuration that was kept, fitted on train
    assert final.named_steps["model"].max_depth == (5 if expected == "tuned" else None)


def never_called(*args, **kwargs):
    raise AssertionError("the search should have been reused from the cache")


@pytest.mark.parametrize("change, reused", [
    ({}, True),                                # identical search -> cache reused
    ({"data_sha256": "other-hash"}, False),     # different data -> new search
    ({"retune": True}, False),                 # forced -> new search
])
def test_tuning_cache(monkeypatch, tmp_path, rf_selection, change, reused):
    split, candidates, selection = rf_selection
    cache = tmp_path / "tuning.json"
    monkeypatch.setattr(train, "tune_model", fake_tuning(0.99))
    first, _, _ = run_tuning(selection, candidates, split, "hash", cache_path=cache)
    cache.write_text(json.dumps(first))

    monkeypatch.setattr(train, "tune_model", never_called if reused else fake_tuning(0.99))
    second, tuned, _ = run_tuning(selection, candidates, split, change.get("data_sha256", "hash"),
                               retune=change.get("retune", False), cache_path=cache)
    assert second["reused_cached_search"] is reused
    assert second["best_params"] == first["best_params"]
    assert tuned["validation"]["pr_auc"] >= 0  # downstream evaluation always recomputed
