"""One-time final test evaluation (SPEC §5.1, §10.1, §11). Synthetic data and temporary
paths only: the real artifact and reports are never touched."""

import json
import platform

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from src import config, evaluate
from src.calibration import fit_calibrator
from src.data_loader import DataSplit, load_raw_data, split_data, split_summary
from src.evaluate import (
    bootstrap_ci,
    calibration_table,
    risk_band_table,
    run_final_evaluation,
    version_mismatches,
)
from src.explain import risk_band
from src.preprocessing import make_model_pipeline
from src.train import make_estimator, split_xy
from tests.fixtures.make_synthetic_data import write_synthetic_csv

UNUSABLE = pd.DataFrame({"not": ["a", "dataset"]})


@pytest.fixture
def project(monkeypatch, tmp_path):
    """A saved artifact, metadata and split summary for synthetic data, in temporary paths.
    Returns the split with train and validation replaced by unusable frames."""
    split = split_data(load_raw_data(write_synthetic_csv(tmp_path / "cs-training.csv", n_rows=1000)))
    X_train, y_train = split_xy(split.train)
    X_val, y_val = split_xy(split.validation)
    pipeline = make_model_pipeline("random_forest", make_estimator("random_forest", n_jobs=1)
                                   .set_params(n_estimators=30)).fit(X_train, y_train)
    artifact = fit_calibrator(pipeline, X_val, y_val, "isotonic")

    paths = {"PIPELINE_PATH": tmp_path / "models" / "pipeline.joblib",
             "METADATA_PATH": tmp_path / "models" / "metadata.json",
             "SPLIT_SUMMARY_PATH": tmp_path / "split_summary.json",
             "TEST_METRICS_PATH": tmp_path / "test_metrics.json",
             "TEST_CALIBRATION_FIGURE": tmp_path / "figures" / "calibration.png",
             "TEST_ROC_PR_FIGURE": tmp_path / "figures" / "roc_pr.png",
             "TEST_CONFUSION_FIGURE": tmp_path / "figures" / "confusion.png"}
    for name, path in paths.items():
        monkeypatch.setattr(config, name, path)
    paths["PIPELINE_PATH"].parent.mkdir()
    joblib.dump(artifact, paths["PIPELINE_PATH"])
    paths["METADATA_PATH"].write_text(json.dumps({
        "model": {"type": "random_forest", "configuration": "untuned"},
        "calibration_method": "isotonic",
        "decision_threshold": 0.17,
        "cost_false_negative": 5.0,
        "cost_false_positive": 1.0,
        "data": {"sha256": "hash"},
        "validation_metrics": {"note": "reference only"},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }))
    paths["SPLIT_SUMMARY_PATH"].write_text(json.dumps(split_summary(split, "hash")))
    test_only = DataSplit(train=UNUSABLE, validation=UNUSABLE, test=split.test)
    return test_only, artifact, paths


def test_final_evaluation_report(project):
    split, artifact, paths = project
    report = run_final_evaluation(split, "hash")

    X, y = split_xy(split.test)
    proba = artifact.predict_proba(X)[:, 1]
    assert report["calibrated"]["pr_auc"] == pytest.approx(evaluate.average_precision_score(y, proba))
    raw = artifact.calibrated_classifiers_[0].estimator.predict_proba(X)[:, 1]
    assert report["raw"]["brier"] == pytest.approx(evaluate.brier_score_loss(y, raw))  # the uncalibrated model

    at = report["at_threshold"]
    assert at["threshold"] == 0.17
    assert at["tp"] + at["fp"] + at["fn"] + at["tn"] == len(y)
    assert at["tp"] + at["fp"] == int((proba >= 0.17).sum())

    for name in ("calibrated", "raw"):
        for metric in ("pr_auc", "roc_auc", "brier"):
            ci = report["bootstrap"][name][metric]
            assert ci["lower"] <= ci["upper"] and ci["std"] > 0
    assert report["bootstrap"]["resamples"] == config.BOOTSTRAP_RESAMPLES

    assert report["test_split"]["matches_recorded_split"] and report["test_split"]["rows"] == len(y)
    assert sum(b["rows"] for b in report["risk_bands"]) == len(y)
    assert report["forced_rerun"] is False and report["version_mismatches"] == {}
    assert report["validation_reference"] == {"note": "reference only"}

    assert json.loads(paths["TEST_METRICS_PATH"].read_text()) == json.loads(json.dumps(report))
    for name in ("TEST_CALIBRATION_FIGURE", "TEST_ROC_PR_FIGURE", "TEST_CONFUSION_FIGURE"):
        assert paths[name].stat().st_size > 0


def test_second_run_is_refused_without_force(project):
    split, _, paths = project
    run_final_evaluation(split, "hash")
    first = paths["TEST_METRICS_PATH"].read_bytes()
    with pytest.raises(FileExistsError, match="--force"):
        run_final_evaluation(split, "hash")
    assert paths["TEST_METRICS_PATH"].read_bytes() == first  # untouched

    forced = run_final_evaluation(split, "hash", force=True)
    assert forced["forced_rerun"] is True
    assert forced["command"].endswith("--force")


def test_force_without_previous_results_is_not_a_rerun(project):
    split, _, _ = project
    assert run_final_evaluation(split, "hash", force=True)["forced_rerun"] is False


def test_refuses_different_data(project):
    split, _, paths = project
    with pytest.raises(ValueError, match="SHA-256"):
        run_final_evaluation(split, "other-hash")
    assert not paths["TEST_METRICS_PATH"].exists()


def test_refuses_a_different_test_split(project):
    split, _, paths = project
    other = DataSplit(train=UNUSABLE, validation=UNUSABLE, test=split.test.iloc[1:])
    with pytest.raises(ValueError, match="row-ID hash"):
        run_final_evaluation(other, "hash")
    assert not paths["TEST_METRICS_PATH"].exists()


def test_requires_the_artifact(project):
    split, _, paths = project
    paths["PIPELINE_PATH"].unlink()
    with pytest.raises(FileNotFoundError, match="src.train"):
        run_final_evaluation(split, "hash")


def test_cli_needs_final_flag():
    with pytest.raises(SystemExit):
        evaluate.main([])


def test_cli_refuses_before_loading_data(project, monkeypatch):
    split, _, paths = project
    paths["TEST_METRICS_PATH"].write_text("{}")

    def must_not_load(*args, **kwargs):
        raise AssertionError("the data must not be loaded when results already exist")
    monkeypatch.setattr("src.data_loader.load_raw_data", must_not_load)
    with pytest.raises(SystemExit, match="already exists"):
        evaluate.main(["--final"])
    assert paths["TEST_METRICS_PATH"].read_text() == "{}"


# --- building blocks ------------------------------------------------------------------------


@pytest.fixture
def calibrated_scores():
    rng = np.random.default_rng(0)
    proba = rng.beta(1, 12, 5000)
    y = (rng.random(5000) < proba).astype(int)  # perfectly calibrated by construction
    return y, proba


def test_bootstrap_is_reproducible_and_covers_the_estimate(calibrated_scores):
    y, proba = calibrated_scores
    noisy = np.clip(proba + np.random.default_rng(1).normal(0, 0.05, len(proba)), 0, 1)
    a = bootstrap_ci(y, {"model": proba, "noisy": noisy}, n_resamples=200)
    b = bootstrap_ci(y, {"model": proba, "noisy": noisy}, n_resamples=200)
    assert a == b  # fixed seed
    for metric, fn in evaluate.BOOTSTRAP_METRICS.items():
        assert a["model"][metric]["lower"] <= fn(y, proba) <= a["model"][metric]["upper"]
    assert a["noisy"]["brier"]["lower"] > a["model"]["brier"]["upper"] - 0.01  # paired sets differ as expected


def test_bootstrap_skips_single_class_resamples():
    result = bootstrap_ci([0, 0, 0, 1], {"m": [0.1, 0.2, 0.3, 0.9]}, n_resamples=200)
    assert result["skipped_single_class"] > 0
    assert result["skipped_single_class"] < 200


def test_calibration_table(calibrated_scores):
    y, proba = calibrated_scores
    table = calibration_table(y, proba, n_bins=5)
    assert len(table) == 5
    predicted = [r["mean_predicted"] for r in table]
    assert predicted == sorted(predicted)
    for r in table:
        assert r["observed_rate"] == pytest.approx(r["mean_predicted"], abs=0.03)


def test_risk_band_table():
    y = [0, 1, 0, 1, 1]
    proba = [0.05, 0.08, 0.2, 0.5, 0.9]
    rows = {r["band"]: r for r in risk_band_table(y, proba)}
    assert [rows[b]["rows"] for b in ("LOW", "MEDIUM", "HIGH")] == [2, 1, 2]
    assert rows["HIGH"]["observed_rate"] == 1.0 and rows["LOW"]["defaults"] == 1
    assert all(risk_band(p) == b for b, p in (("LOW", 0.05), ("MEDIUM", 0.2), ("HIGH", 0.9)))


def test_risk_band_table_handles_empty_bands():
    rows = {r["band"]: r for r in risk_band_table([0, 1], [0.01, 0.02])}
    assert rows["HIGH"]["rows"] == 0 and rows["HIGH"]["observed_rate"] is None


def test_pr_chart_baseline_is_labelled_in_the_legend(calibrated_scores, tmp_path):
    """The default-rate label once sat on the baseline and overlapped the curve's end."""
    y, proba = calibrated_scores
    fig = evaluate.plot_roc_pr_curves(y, proba, proba, 0.17, tmp_path / "roc_pr.png")
    pr_ax = fig.axes[1]
    legend = [t.get_text() for t in pr_ax.get_legend().get_texts()]
    assert any(t.startswith("No-skill baseline (default rate") for t in legend)
    assert not any("default rate" in t.get_text() for t in pr_ax.texts)  # no free-floating label
    assert (tmp_path / "roc_pr.png").stat().st_size > 0


def test_version_mismatches():
    assert version_mismatches({"scikit-learn": sklearn.__version__}) == {}
    assert version_mismatches({"scikit-learn": "0.1"}) == {
        "scikit-learn": {"recorded": "0.1", "installed": sklearn.__version__}}
