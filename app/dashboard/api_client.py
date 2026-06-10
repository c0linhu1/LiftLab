"""
api_client.py — Thin wrapper around the LiftLab FastAPI backend.

Streamlit pages call these helpers instead of building HTTP calls
inline. Cached at the Streamlit layer so repeated visits to the same
page don't re-hit slow endpoints (segmentation, decision, CUPED).
"""

import os
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 120


def _get(path: str, params: dict | None = None) -> dict:
    """GET <API_URL><path> and return parsed JSON. Raises on non-2xx."""
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=600, show_spinner=False)
def health() -> dict:
    return _get("/health")


@st.cache_data(ttl=600, show_spinner=False)
def list_experiments() -> list:
    return _get("/experiments")


@st.cache_data(ttl=600, show_spinner=False)
def get_experiment(experiment_id: str) -> dict:
    return _get(f"/experiments/{experiment_id}")


@st.cache_data(ttl=600, show_spinner=False)
def topline() -> dict:
    return _get("/topline")


@st.cache_data(ttl=600, show_spinner=False)
def srm() -> dict:
    return _get("/srm")


@st.cache_data(ttl=600, show_spinner=False)
def frequentist(analysis_unit: str = "user") -> dict:
    return _get("/frequentist", params={"analysis_unit": analysis_unit})


@st.cache_data(ttl=600, show_spinner=False)
def cuped() -> dict:
    return _get("/cuped")


@st.cache_data(ttl=600, show_spinner=False)
def bayesian(analysis_unit: str = "user") -> dict:
    return _get("/bayesian", params={"analysis_unit": analysis_unit})


@st.cache_data(ttl=600, show_spinner=False)
def segments() -> dict:
    return _get("/segments")


@st.cache_data(ttl=600, show_spinner=False)
def guardrails() -> dict:
    return _get("/guardrails")


@st.cache_data(ttl=600, show_spinner=False)
def decision() -> dict:
    return _get("/decision")


def banner_api_status() -> None:
    """Render a small connection-status banner at the top of every page."""
    try:
        h = health()
        if h.get("data_loaded"):
            st.caption(
                f"API: connected · {h.get('n_users', 0):,} users · "
                f"{h.get('n_sessions', 0):,} sessions"
            )
        else:
            st.warning("API is up but data is still loading. Refresh in a few seconds.")
    except Exception as e:
        st.error(f"Cannot reach API at {API_URL}. Is the backend running?  ({e})")
