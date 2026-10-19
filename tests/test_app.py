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


def _cid_input(at: AppTest):
    return at.text_input(key="cid")


def _all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_customer_lookup_accepts_valid_id() -> None:
    at = _run_app()
    _cid_input(at).set_value("14688")
    at.run()
    assert not at.exception
    assert "Enter a valid numeric" not in _warnings_text(at)
    assert "not found in dataset" not in _warnings_text(at)
    assert "Customer 14688 vs all customers" in _all_markdown(at)


def test_customer_lookup_rejects_non_numeric_id() -> None:
    at = _run_app()
    _cid_input(at).set_value("not-a-number")
    at.run()
    assert "Enter a valid numeric" in _warnings_text(at)


def test_customer_lookup_unknown_id() -> None:
    at = _run_app()
    _cid_input(at).set_value("1")
    at.run()
    assert "not found in dataset" in _warnings_text(at)


def test_default_view_includes_every_customer() -> None:
    """Regression: int(max spend) truncated the slider and silently dropped the top spender."""
    import pandas as pd

    total = len(pd.read_csv(os.path.join(os.path.dirname(APP_PATH), "data", "rfm_scored.csv")))
    at = _run_app()
    assert f"{total:,}" in _all_markdown(at)
    assert f"{total:,}</div>" in _all_markdown(at).replace("\n", "")


def test_quick_view_preset_sets_segments_and_shrinks_view() -> None:
    at = _run_app()
    at.sidebar.selectbox(key="quick_view").select("Needs attention")
    at.run()
    assert at.sidebar.multiselect[0].value == ["At Risk", "Lost / Churned"]
    assert "Filtered view" in _all_markdown(at)
    assert not at.exception


def test_reset_filters_restores_full_view() -> None:
    at = _run_app()
    at.sidebar.selectbox(key="quick_view").select("Growth pool")
    at.run()
    at.sidebar.button(key="reset_filters").click()
    at.run()
    assert len(at.sidebar.multiselect[0].value) == 5
    assert "Filtered view" not in _all_markdown(at)


def test_score_filter_updates_without_error() -> None:
    at = _run_app()
    at.sidebar.slider(key="f_score").set_value((12, 15))
    at.run()
    assert not at.exception
    assert "Filtered view" in _all_markdown(at)


def test_export_download_button_present() -> None:
    at = _run_app()
    assert len(at.download_button) >= 1


def test_every_deep_dive_segment_renders() -> None:
    from constants import SEGMENT_ORDER

    at = _run_app()
    for name in SEGMENT_ORDER:
        at.selectbox(key="deep_seg").select(name)
        at.run()
        assert not at.exception, name


def test_every_simulator_segment_and_edge_inputs() -> None:
    from constants import SEGMENT_ORDER

    at = _run_app()
    for name in SEGMENT_ORDER:
        at.selectbox(key="sim_seg").select(name)
        at.run()
        assert not at.exception, name
    at.number_input(key="sim_cost").set_value(100.0)  # cost > profit per responder
    at.run()
    assert not at.exception
    at.number_input(key="sim_cost").set_value(0.0)  # zero cost -> ROI undefined
    at.run()
    assert not at.exception


def test_customer_outside_filters_still_shown() -> None:
    at = _run_app()
    at.sidebar.selectbox(key="quick_view").select("Growth pool")
    at.run()
    _cid_input(at).set_value("14688")  # a Champion, outside the Growth pool view
    at.run()
    assert not at.exception
    assert "Customer 14688 vs all customers" in _all_markdown(at)


def test_assess_roi_priority_is_not_title_cased() -> None:
    import pandas as pd

    df = pd.read_csv(os.path.join(os.path.dirname(APP_PATH), "data", "rfm_scored.csv"))
    cid = int(df.loc[df["Segment"] == "Lost / Churned", "CustomerID"].iloc[0])
    at = _run_app()
    _cid_input(at).set_value(str(cid))
    at.run()
    md = _all_markdown(at)
    assert "Assess ROI" in md and "Assess Roi" not in md


def test_dollar_currency_does_not_break_markdown(monkeypatch=None) -> None:
    import ui

    assert "$" not in ui.kpi_card("Revenue", "$8.35M", "<b>35%</b> of $8.35M")
