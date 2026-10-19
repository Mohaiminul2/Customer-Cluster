"""Smoke tests: every chart builder returns a populated figure, including edge cases."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics import segment_profile, summarize_segments
from charts import (
    band_heatmap,
    customer_map,
    distribution_figure,
    economics_waterfall,
    k_diagnostics,
    pareto_figure,
    percentile_bars,
    profile_heatmap,
    score_distribution,
    share_bars,
)
from clustering import diagnostics_table
from config_loader import BASE_DIR, load_config

CFG = load_config()


@pytest.fixture(scope="module")
def rfm() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE_DIR, "data", "rfm_scored.csv"))


def _ok(fig: go.Figure) -> None:
    assert isinstance(fig, go.Figure) and len(fig.data) > 0


def test_overview_charts(rfm: pd.DataFrame) -> None:
    h = CFG["heatmap"]
    _ok(share_bars(summarize_segments(rfm)))
    _ok(pareto_figure(rfm["Monetary"]))
    _ok(profile_heatmap(segment_profile(rfm, rfm)))
    for metric in ("customers", "revenue"):
        _ok(band_heatmap(rfm, h["recency_bins"], h["recency_labels"],
                         h["monetary_bins"], h["monetary_labels"], metric))


@pytest.mark.parametrize("metric", ["Recency", "Frequency", "Monetary"])
def test_distribution_charts(rfm: pd.DataFrame, metric: str) -> None:
    seg = rfm[rfm["Segment"] == "Champions"]
    _ok(distribution_figure(seg[metric], rfm[metric], metric, "#34D399"))


def test_single_customer_view_does_not_crash(rfm: pd.DataFrame) -> None:
    one = rfm.iloc[[0]]
    _ok(share_bars(summarize_segments(one)))
    _ok(pareto_figure(one["Monetary"]))
    _ok(profile_heatmap(segment_profile(one, rfm)))
    _ok(customer_map(one, one.iloc[0]))


def test_explorer_and_method_charts(rfm: pd.DataFrame) -> None:
    _ok(customer_map(rfm, rfm.iloc[0]))
    _ok(percentile_bars({"Recency": 90.0, "Frequency": 50.0, "Monetary": 99.0}, "#34D399"))
    _ok(score_distribution(rfm))
    diag = diagnostics_table(rfm, CFG)
    assert list(diag["K"]) == list(range(CFG["clustering"]["k_range_start"],
                                         CFG["clustering"]["k_range_end"] + 1))
    _ok(k_diagnostics(diag, CFG["clustering"]["k_final"]))


def test_waterfall() -> None:
    _ok(economics_waterfall({"gross_profit": 500.0, "cost": 200.0, "net_profit": 300.0}))
