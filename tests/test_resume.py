"""`python -m src.train --from-phase 9`: resume after Phase 8 without rerunning or rewriting it.
Synthetic data only."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config, train
from src.data_loader import DataSplit, load_raw_data, split_data
from src.train import load_phase_8_results, run_candidates, run_tuning, select_model, split_xy, write_json
from tests.fixtures.make_synthetic_data import write_synthetic_csv
from tests.test_tune import fake_tuning


@pytest.fixture
def phase_8(monkeypatch, tmp_path):
    """Saved Phase 6-8 results for a synthetic split, as `python -m src.train` writes them."""
    split = split_data(load_raw_data(write_synthetic_csv(tmp_path / "cs-training.csv", n_rows=1000)))
    # Replace the test split with something unusable: any access would raise.
    split = DataSplit(train=split.train, validation=split.validation, test=pd.DataFrame({"not": ["a", "dataset"]}))

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "tune_model", fake_tuning(0.99))
    candidates = run_candidates(split, models=("random_forest",))
    selection = select_model(candidates)
    tuning, tuned_result, _ = run_tuning(selection, candidates, split, "hash", cache_path=tmp_path / "unused.json")
    selection["final_configuration"] = tuning["final_configuration"]

    paths = {"tuning_path": tmp_path / "tuning_results.json", "comparison_path": tmp_path / "model_comparison.json"}
    write_json(paths["tuning_path"], tuning)
    write_json(paths["comparison_path"], {"data_sha256": "hash", "candidates": [*candidates, tuned_result],
                                          "selection": selection})
    return split, paths


def test_resume_refits_the_saved_configuration(phase_8):
    split, paths = phase_8
    selection, tuning, final = load_phase_8_results(split, "hash", **paths)
    assert tuning["final_configuration"] == "tuned"
    assert selection["selected_model"] == "random_forest"
    assert final.named_steps["model"].max_depth == 5  # the tuned parameters, fitted on train
    assert final.named_steps["model"].n_estimators == 20


def test_resume_rejects_different_data(phase_8):
    split, paths = phase_8
    with pytest.raises(ValueError, match="different data"):
        load_phase_8_results(split, "other-hash", **paths)


def test_resume_rejects_changed_search(phase_8):
    split, paths = phase_8
    tuning = json.loads(paths["tuning_path"].read_text())
    tuning["signature"]["scikit_learn_version"] = "0.0"
    paths["tuning_path"].write_text(json.dumps(tuning))
    with pytest.raises(ValueError, match="does not match"):
        load_phase_8_results(split, "hash", **paths)


def test_resume_rejects_unreproducible_model(phase_8):
    split, paths = phase_8
    comparison = json.loads(paths["comparison_path"].read_text())
    comparison["candidates"][-1]["validation"]["pr_auc"] += 0.001
    paths["comparison_path"].write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="reproduced|saved"):
        load_phase_8_results(split, "hash", **paths)


def test_resume_requires_phase_8_results(phase_8, tmp_path):
    split, paths = phase_8
    with pytest.raises(FileNotFoundError, match="without --from-phase"):
        load_phase_8_results(split, "hash", **{**paths, "tuning_path": tmp_path / "missing.json"})


def test_from_phase_cannot_be_combined_with_retune():
    with pytest.raises(SystemExit):
        train.main(["--from-phase", "9", "--retune"])


def test_from_phase_9_end_to_end(monkeypatch, tmp_path, phase_8):
    split, paths = phase_8
    for name, path in (("TUNING_RESULTS_PATH", paths["tuning_path"]),
                       ("MODEL_COMPARISON_PATH", paths["comparison_path"]),
                       ("CALIBRATION_RESULTS_PATH", tmp_path / "calibration_results.json"),
                       ("THRESHOLD_ANALYSIS_PATH", tmp_path / "threshold_analysis.json"),
                       ("FIGURES_DIR", tmp_path / "figures"),
                       ("MODELS_DIR", tmp_path / "models"),
                       ("PIPELINE_PATH", tmp_path / "models" / "pipeline.joblib"),
                       ("METADATA_PATH", tmp_path / "models" / "metadata.json")):
        monkeypatch.setattr(config, name, path)
    monkeypatch.setattr(train, "file_sha256", lambda path: "hash")
    monkeypatch.setattr(train, "load_raw_data", lambda: None)
    monkeypatch.setattr(train, "split_data", lambda df: split)

    def must_not_run(*args, **kwargs):
        raise AssertionError("earlier phases must not be rerun")
    monkeypatch.setattr(train, "run_candidates", must_not_run)
    monkeypatch.setattr(train, "run_tuning", must_not_run)
    before = {p: p.read_bytes() for p in paths.values()}

    train.main(["--from-phase", "9"])

    assert {p: p.read_bytes() for p in paths.values()} == before  # Phase 6-8 reports untouched
    metadata = json.loads(config.METADATA_PATH.read_text())
    threshold = json.loads(config.THRESHOLD_ANALYSIS_PATH.read_text())["chosen_threshold"]
    assert metadata["decision_threshold"] == threshold
    assert metadata["model"]["configuration"] == "tuned"
    assert metadata["model"]["hyperparameters"]["max_depth"] == 5
    assert (config.FIGURES_DIR / "threshold_analysis_validation.png").exists()

    loaded = joblib.load(config.PIPELINE_PATH)
    pd_val = loaded.predict_proba(split_xy(split.validation)[0])[:, 1]
    assert np.all((pd_val >= 0) & (pd_val <= 1))
