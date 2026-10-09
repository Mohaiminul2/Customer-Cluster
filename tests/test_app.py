"""Tests for the Streamlit dashboard (app.py)."""

from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)


def _run_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def _warnings_text(at: AppTest) -> str:
    return "\n".join(w.value for w in at.warning)


def test_app_loads_without_exception() -> None:
    at = _run_app()
    assert not at.exception


def test_empty_state_shows_warning() -> None:
    at = _run_app()
    at.sidebar.multiselect[0].set_value([])
    at.run()
    assert "No customers match" in _warnings_text(at)


def test_customer_lookup_accepts_valid_id() -> None:
    at = _run_app()
    at.sidebar.text_input[0].set_value("14688")
    at.run()
    assert "Enter a valid numeric" not in _warnings_text(at)
    assert "not found in dataset" not in _warnings_text(at)


def test_customer_lookup_rejects_non_numeric_id() -> None:
    at = _run_app()
    at.sidebar.text_input[0].set_value("not-a-number")
    at.run()
    assert "Enter a valid numeric" in _warnings_text(at)


def test_export_download_button_present() -> None:
    at = _run_app()
    assert len(at.download_button) >= 1
