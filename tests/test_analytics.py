"""Tests for the pure analytics helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from analytics import (
    band_matrix,
    build_playbook,
    campaign_economics,
    customer_percentiles,
    fmt_count,
    fmt_money,
    generate_insights,
    lorenz_curve,
    segment_profile,
    summarize_segments,
    target_customers,
    top_share,
)
from config_loader import BASE_DIR, load_config
from constants import SEGMENT_ORDER

CFG = load_config()
PARAMS = CFG["playbook"]


@pytest.fixture(scope="module")
def real_rfm() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE_DIR, "data", "rfm_scored.csv"))


class TestFormatting:
    @pytest.mark.parametrize("value,expected", [
        (842, "£842"), (4310, "£4.3k"), (25977, "£26k"), (3326066, "£3.33M"), (-1500, "-£1.5k"),
    ])
    def test_fmt_money(self, value: float, expected: str) -> None:
        assert fmt_money(value) == expected

    def test_fmt_count(self) -> None:
        assert fmt_count(4338) == "4,338"
        assert fmt_count(4.3) == "4.3"


class TestSummary:
    def test_matches_stored_pipeline_summary(self, real_rfm: pd.DataFrame) -> None:
        stored = pd.read_csv(os.path.join(BASE_DIR, "data", "segment_summary.csv"))
        mine = summarize_segments(real_rfm)
        assert list(mine["Segment"]) == list(stored["Segment"])
        for col in ("Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary",
                    "Total_Revenue", "Revenue_Pct"):
            assert np.allclose(mine[col].round(1), stored[col], atol=0.11), col

    def test_summary_follows_filters(self, real_rfm: pd.DataFrame) -> None:
        """Regression: Overview used the static CSV and ignored score/spend filters."""
        view = real_rfm[real_rfm["RFM_Score"] >= 12]
        s = summarize_segments(view)
        assert s["Customers"].sum() == len(view)
        assert s["Total_Revenue"].sum() == pytest.approx(view["Monetary"].sum())
        assert s["Revenue_Pct"].sum() == pytest.approx(100.0)

    def test_empty_frame(self, real_rfm: pd.DataFrame) -> None:
        assert summarize_segments(real_rfm.iloc[0:0]).empty

    def test_order_value_is_revenue_per_order(self, real_rfm: pd.DataFrame) -> None:
        s = summarize_segments(real_rfm).set_index("Segment")
        champs = real_rfm[real_rfm["Segment"] == "Champions"]
        assert s.loc["Champions", "Avg_Order_Value"] == pytest.approx(
            champs["Monetary"].sum() / champs["Frequency"].sum())


class TestConcentration:
    def test_lorenz_bounds(self) -> None:
        x, y = lorenz_curve(pd.Series([1, 2, 3, 94]))
        assert (x[0], y[0]) == (0, 0) and (x[-1], y[-1]) == (1, pytest.approx(1))
        assert y[1] == pytest.approx(0.94)  # biggest spender first

    def test_top_share(self) -> None:
        assert top_share(pd.Series([50, 30, 10, 10]), 0.25) == pytest.approx(0.5)
        assert top_share(pd.Series([], dtype=float), 0.2) == 0.0

    def test_equal_spend_is_proportional(self) -> None:
        assert top_share(pd.Series([10] * 10), 0.2) == pytest.approx(0.2)


class TestProfileAndBands:
    def test_profile_sign_means_better(self, real_rfm: pd.DataFrame) -> None:
        p = segment_profile(real_rfm)
        assert (p.loc["Champions", ["Recency_score", "Frequency_score", "Monetary_score"]] > 0).all()
        assert (p.loc["Lost / Churned", "Recency_score"]) < 0
        assert p[["Recency_score", "Frequency_score", "Monetary_score"]].abs().max().max() <= 3

    def test_band_matrix_covers_every_customer(self, real_rfm: pd.DataFrame) -> None:
        h = CFG["heatmap"]
        grid = band_matrix(real_rfm, h["recency_bins"], h["recency_labels"],
                           h["monetary_bins"], h["monetary_labels"])
        assert grid.to_numpy().sum() == len(real_rfm)

    def test_band_matrix_revenue_sums_to_100(self, real_rfm: pd.DataFrame) -> None:
        h = CFG["heatmap"]
        grid = band_matrix(real_rfm, h["recency_bins"], h["recency_labels"],
                           h["monetary_bins"], h["monetary_labels"], metric="revenue")
        assert grid.to_numpy().sum() == pytest.approx(100.0)

    def test_band_matrix_survives_narrow_view(self, real_rfm: pd.DataFrame) -> None:
        """Regression: old code patched the last bin edge by hand when the view was narrow."""
        h = CFG["heatmap"]
        view = real_rfm[(real_rfm["Recency"] <= 10) & (real_rfm["Monetary"] < 400)]
        grid = band_matrix(view, h["recency_bins"], h["recency_labels"],
                           h["monetary_bins"], h["monetary_labels"])
        assert grid.to_numpy().sum() == len(view)

    def test_band_label_mismatch_raises(self, real_rfm: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            band_matrix(real_rfm, [0, 30], ["a"], [0, 500], ["x", "y"])


class TestInsightsAndPlaybook:
    def test_insights_are_grounded_in_data(self, real_rfm: pd.DataFrame) -> None:
        ins = generate_insights(real_rfm, summarize_segments(real_rfm), "£", PARAMS)
        text = " ".join(i.headline for i in ins)
        t20 = round(top_share(real_rfm["Monetary"], 0.2) * 100)
        assert f"{t20}%" in text
        assert any("At Risk" in i.headline for i in ins)

    def test_insights_single_segment_does_not_crash(self, real_rfm: pd.DataFrame) -> None:
        one = real_rfm[real_rfm["Segment"] == "Champions"]
        assert generate_insights(one, summarize_segments(one), "£", PARAMS)

    def test_playbook_sizes(self, real_rfm: pd.DataFrame) -> None:
        pb = build_playbook(real_rfm, real_rfm, PARAMS).set_index("Segment")
        at_risk = real_rfm[real_rfm["Segment"] == "At Risk"]
        assert pb.loc["At Risk", "Targeted customers"] == (at_risk["Frequency"] >= 3).sum()
        lost = real_rfm[real_rfm["Segment"] == "Lost / Churned"]
        assert pb.loc["Lost / Churned", "Targeted customers"] == int(np.ceil(len(lost) * 0.25))
        assert (pb["Targeted customers"] <= pb["Segment customers"]).all()
        assert list(pb.index) == SEGMENT_ORDER

    def test_target_lists_belong_to_segment_and_sorted(self, real_rfm: pd.DataFrame) -> None:
        for seg in SEGMENT_ORDER:
            t = target_customers(real_rfm, real_rfm, seg, PARAMS)
            assert (t["Segment"] == seg).all()
            assert t["Monetary"].is_monotonic_decreasing

    def test_potential_loyalists_rule(self, real_rfm: pd.DataFrame) -> None:
        t = target_customers(real_rfm, real_rfm, "Potential Loyalists", PARAMS)
        loyal_med = real_rfm.loc[real_rfm["Segment"] == "Loyal Customers", "Frequency"].median()
        assert (t["Recency"] <= PARAMS["potential_recent_days"]).all()
        assert (t["Frequency"] < loyal_med).all()


class TestEconomics:
    def test_breakeven_gives_zero_net(self) -> None:
        e = campaign_economics(1000, 100.0, 0.05, 2.0, 0.30, 0.10)
        be = e["breakeven_response"]
        assert be == pytest.approx(2.0 / (100 * 0.20))
        at_be = campaign_economics(1000, 100.0, be, 2.0, 0.30, 0.10)
        assert at_be["net_profit"] == pytest.approx(0.0, abs=1e-9)

    def test_known_numbers(self) -> None:
        e = campaign_economics(1000, 100.0, 0.10, 1.0, 0.30, 0.10)
        assert e["responders"] == pytest.approx(100)
        assert e["gross_profit"] == pytest.approx(100 * 100 * 0.20)
        assert e["cost"] == pytest.approx(1000)
        assert e["net_profit"] == pytest.approx(1000)
        assert e["roi"] == pytest.approx(1.0)

    def test_discount_above_margin_never_profitable(self) -> None:
        e = campaign_economics(100, 50.0, 0.5, 1.0, 0.20, 0.30)
        assert e["breakeven_response"] is None and e["net_profit"] < 0

    def test_zero_cost_roi_is_none(self) -> None:
        assert campaign_economics(100, 50.0, 0.1, 0.0, 0.3, 0.1)["roi"] is None


def test_customer_percentiles(real_rfm: pd.DataFrame) -> None:
    top = real_rfm.loc[real_rfm["Monetary"].idxmax()]
    p = customer_percentiles(top, real_rfm)
    assert p["Monetary"] > 99.9
    worst = real_rfm.loc[real_rfm["Recency"].idxmax()]
    assert customer_percentiles(worst, real_rfm)["Recency"] < 1
