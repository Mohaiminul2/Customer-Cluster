"""Tests for config loading."""

from __future__ import annotations

import os

from rfm_analysis import _load_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_loads() -> None:
    cfg = _load_config()
    assert "clustering" in cfg
    assert "caps" in cfg
    assert "heatmap" in cfg


def test_clustering_config_keys() -> None:
    cfg = _load_config()
    cc = cfg["clustering"]
    assert isinstance(cc["k_final"], int)
    assert isinstance(cc["random_state"], int)
    assert isinstance(cc["n_init"], int)
    assert cc["k_range_start"] < cc["k_range_end"]


def test_caps_are_percentiles() -> None:
    cfg = _load_config()
    for key in ("monetary_quantile", "frequency_quantile", "recency_quantile"):
        val = cfg["caps"][key]
        assert 0.0 < val < 1.0, f"{key}={val} is not a valid percentile"


def test_heatmap_bins_and_labels_match() -> None:
    cfg = _load_config()
    assert len(cfg["heatmap"]["recency_bins"]) == len(cfg["heatmap"]["recency_labels"])
    assert len(cfg["heatmap"]["monetary_bins"]) == len(cfg["heatmap"]["monetary_labels"])
