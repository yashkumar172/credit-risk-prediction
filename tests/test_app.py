"""Streamlit app tests with AppTest (SPEC §14, §19.5, §19.6): headless, in-process.

The app runs against a synthetic artifact in a temporary folder, so these tests need
neither the Kaggle data nor the trained model. Tests against the real artifact and
reports are skipped when those are absent.
"""

import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from tests.fixtures.make_synthetic_artifact import write_synthetic_artifact

APP = str(config.PROJECT_ROOT / "app.py")
PAGES = ["app_pages/home.py", "app_pages/prediction.py", "app_pages/performance.py",
         "app_pages/explainability.py", "app_pages/about.py"]
REAL_METRICS_DIR, REAL_FIGURES_DIR = config.METRICS_DIR, config.FIGURES_DIR


def never(*args, **kwargs):
    raise AssertionError("the app must not read the raw dataset")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return write_synthetic_artifact(tmp_path_factory.mktemp("artifact"))


@pytest.fixture
def app_env(monkeypatch, tmp_path, synthetic):
    """The synthetic model and empty report folders; caches cleared around each test."""
    pipeline_path, metadata_path, metadata = synthetic
    monkeypatch.setattr(config, "PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(config, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr("src.data_loader.load_raw_data", never)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield metadata
    st.cache_resource.clear()
    st.cache_data.clear()


def open_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_reports(app_env, page):
    at = open_page(page)
    assert at.title[0].value  # the page title from st.navigation
    if page in ("app_pages/performance.py", "app_pages/explainability.py"):
        assert at.info  # explains that the reports have not been produced yet


def test_app_inputs_match_metadata_exactly_in_order(app_env):
    """SPEC §19.5: the input fields are the metadata's raw feature list, in order."""
    at = open_page("app_pages/prediction.py")
    assert [n.key for n in at.number_input] == app_env["raw_input_features"]
    assert [n.label for n in at.number_input] == [config.FEATURE_LABELS[f] for f in app_env["raw_input_features"]]
    unknown = [c.key for c in at.checkbox]
    assert unknown == ["MonthlyIncome__unknown", "NumberOfDependents__unknown"]  # SPEC §14.2


def test_hard_limits_on_inputs(app_env):
    at = open_page("app_pages/prediction.py")
    age = at.number_input(key="age")
    assert (age.min, age.max) == (18, 110)
    assert all(n.min == 0 for n in at.number_input if n.key != "age")  # no negative values


def test_prediction_end_to_end(app_env):
    """SPEC §19.6 through the app: raw input → PD → band → flag → SHAP explanation."""
    at = open_page("app_pages/prediction.py")
    assert not at.metric  # nothing before the button
    at.button[0].click().run()
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert set(metrics) == {"Probability of default", "Risk band", "Flagged for manual review"}
    pd_value = float(metrics["Probability of default"].rstrip("%")) / 100
    assert metrics["Risk band"] in ("LOW", "MEDIUM", "HIGH")
    assert metrics["Flagged for manual review"] == ("Yes" if pd_value >= app_env["decision_threshold"] else "No")

    text = "\n".join(m.value for m in at.markdown)
    assert "Factors increasing risk" in text and "Factors reducing risk" in text
    captions = "\n".join(c.value for c in at.caption)
    assert "do not add up exactly to the calibrated PD" in captions  # SPEC §13.1 note
    assert f"Decision threshold: {app_env['decision_threshold']:.2f}" in captions
    assert at.get("vega_lite_chart")  # the contribution chart


def test_changed_inputs_hide_the_old_result(app_env):
    at = open_page("app_pages/prediction.py")
    at.button[0].click().run()
    assert at.metric
    at.number_input(key="age").set_value(70).run()
    assert not at.metric
    assert any("inputs have changed" in i.value for i in at.info)


def test_unknown_income_disables_debt_ratio_and_predicts(app_env):
    at = open_page("app_pages/prediction.py")
    at.checkbox(key="MonthlyIncome__unknown").check().run()
    assert at.number_input(key="MonthlyIncome").disabled
    assert at.number_input(key="DebtRatio").disabled
    at.button[0].click().run()
    assert not at.exception and at.metric
    assert any("ignores the debt ratio" in w.value for w in at.warning)


def test_missing_artifact_shows_a_message_not_a_crash(app_env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PIPELINE_PATH", tmp_path / "missing.joblib")
    at = open_page("app_pages/prediction.py")
    assert any("python -m src.train" in e.value for e in at.error)
    assert not at.number_input


# --- Home page: saved test results only -----------------------------------------------------

# Distinctive made-up values: the page must show exactly these, proving it reads the saved report
FAKE_TEST_METRICS = {
    "calibrated": {"pr_auc": 0.4321, "roc_auc": 0.8766},
    "bootstrap": {"level": 0.95, "resamples": 1000,
                  "calibrated": {"pr_auc": {"lower": 0.4012, "upper": 0.4598},
                                 "roc_auc": {"lower": 0.8611, "upper": 0.8899}}},
    "at_threshold": {"threshold": 0.23, "recall": 0.5555, "flagged_rate": 0.0987},
    "test_split": {"default_rate": 0.0712},
}


def write_test_metrics(metrics: dict) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR / "test_metrics.json").write_text(json.dumps(metrics))


def home_metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def test_home_shows_the_saved_test_results(app_env):
    write_test_metrics(FAKE_TEST_METRICS)
    at = open_page(PAGES[0])
    assert home_metrics(at) == {"PR-AUC": "0.432", "ROC-AUC": "0.877", "Recall at 0.23 threshold": "55.5%",
                                "Borrowers flagged": "9.9%"}
    assert "Defaulters caught" not in home_metrics(at)


def test_home_shows_confidence_intervals_from_the_saved_bootstrap(app_env):
    write_test_metrics(FAKE_TEST_METRICS)
    captions = "\n".join(c.value for c in open_page(PAGES[0]).caption)
    assert "95% confidence intervals (1,000 bootstrap resamples of the test set)" in captions
    assert "PR-AUC 0.401 to 0.460" in captions
    assert "ROC-AUC 0.861 to 0.890" in captions


def test_home_explains_pr_auc_and_the_target(app_env):
    write_test_metrics(FAKE_TEST_METRICS)
    text = "\n".join(m.value for m in open_page(PAGES[0]).markdown)
    assert "Why PR-AUC?" in text
    assert "Only 7.1% of borrowers default" in text and "93% accurate" in text  # from the saved default rate
    assert "scores about 0.071" in text
    assert "90 days or more past due" in text and "within the two years" in text


def test_home_without_test_results_omits_them(app_env):
    at = open_page(PAGES[0])
    assert not at.metric
    assert "Why PR-AUC?" not in "\n".join(m.value for m in at.markdown)
    assert "90 days or more past due" in "\n".join(m.value for m in at.markdown)  # the target is always explained


# --- the real artifact and reports (skipped when absent) --------------------------------------

real = pytest.mark.skipif(not (config.METADATA_PATH.exists() and (REAL_METRICS_DIR / "test_metrics.json").exists()),
                          reason="trained artifact or reports not present")


@real
def test_real_metadata_matches_the_app_inputs(app_env, monkeypatch):
    """SPEC §19.5 against the committed metadata."""
    metadata_path = config.PROJECT_ROOT / "models" / "model_metadata.json"
    monkeypatch.setattr(config, "METADATA_PATH", metadata_path)
    at = open_page("app_pages/prediction.py")
    assert [n.key for n in at.number_input] == json.loads(metadata_path.read_text())["raw_input_features"]


@real
def test_home_matches_the_real_test_metrics(app_env, monkeypatch):
    monkeypatch.setattr(config, "METRICS_DIR", REAL_METRICS_DIR)
    saved = json.loads((REAL_METRICS_DIR / "test_metrics.json").read_text())
    at = open_page(PAGES[0])
    ci, threshold = saved["bootstrap"]["calibrated"], saved["at_threshold"]["threshold"]
    assert home_metrics(at)["PR-AUC"] == f"{saved['calibrated']['pr_auc']:.3f}"
    assert home_metrics(at)["ROC-AUC"] == f"{saved['calibrated']['roc_auc']:.3f}"
    assert home_metrics(at)[f"Recall at {threshold:.2f} threshold"] == f"{saved['at_threshold']['recall']:.1%}"
    captions = "\n".join(c.value for c in at.caption)
    assert f"PR-AUC {ci['pr_auc']['lower']:.3f} to {ci['pr_auc']['upper']:.3f}" in captions
    assert f"ROC-AUC {ci['roc_auc']['lower']:.3f} to {ci['roc_auc']['upper']:.3f}" in captions


@real
@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_the_real_reports(app_env, monkeypatch, page):
    monkeypatch.setattr(config, "METRICS_DIR", REAL_METRICS_DIR)
    monkeypatch.setattr(config, "FIGURES_DIR", REAL_FIGURES_DIR)
    at = open_page(page)
    assert not at.error or page == "app_pages/about.py"  # About shows the "do not use" notice as an error box
    if page == "app_pages/performance.py":
        assert {m.label for m in at.metric} >= {"PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"}
        assert len(at.dataframe) == 3  # test metrics, risk bands, model comparison
        comparison = at.dataframe[2].value
        deployed = comparison[comparison["Deployed"]]
        assert len(deployed) == 1 and deployed["Model"].iloc[0] == "Random Forest (tuned)"
