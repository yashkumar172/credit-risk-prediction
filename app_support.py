"""Cached loaders shared by the app pages. Read-only: models/ and reports/ only."""

from __future__ import annotations

import json

import streamlit as st

from src import config
from src.predict import Predictor, load_predictor

ARTIFACT_MISSING = ("The trained model was not found. Run `python -m src.train` first "
                    "(see the README), then reload this page.")


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor() -> Predictor:
    """The artifact, metadata and SHAP explainer, loaded once per server (SPEC §14.6)."""
    return load_predictor()


def predictor_or_stop() -> Predictor:
    """The predictor, or an error message and an early stop if the artifact is missing."""
    try:
        return get_predictor()
    except FileNotFoundError:
        st.error(ARTIFACT_MISSING, icon=":material/error:")
        st.stop()


@st.cache_data(max_entries=16, show_spinner=False)
def load_report(name: str) -> dict | None:
    """A JSON report from reports/metrics, or None if it has not been produced yet."""
    path = config.METRICS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def show_figure(name: str, alt: str, caption: str | None = None) -> None:
    """A saved figure from reports/figures, or a note if it is missing."""
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, alt=alt)
    else:
        st.info(f"Figure `{name}` has not been produced yet.", icon=":material/image_not_supported:")
