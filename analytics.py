"""Pure analytics helpers for the dashboard (no Streamlit imports).

Everything here takes DataFrames in and returns DataFrames / numbers / strings
out, so it can be unit-tested without a browser. The dashboard derives every
number from ``rfm_scored.csv`` through these functions, which keeps all charts
consistent with the active sidebar filters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from constants import (
    ACCENT,
    AMBER,
    RECOMMENDATIONS,
    SEGMENT_COLORS,
    SEGMENT_ORDER,
)

FEATURES = ["Recency", "Frequency", "Monetary"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_money(value: float, symbol: str = "£") -> str:
    """Compact currency: £842, £4.3k, £26k, £3.33M."""
    sign = "-" if value < 0 else ""
    a = abs(float(value))
    if a >= 1e6 or round(a / 1e3) >= 1000:  # 999.6k would otherwise print as "1000k"
        return f"{sign}{symbol}{a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}{symbol}{a / 1e3:.0f}k"
    if a >= 1e3:
        return f"{sign}{symbol}{a / 1e3:.1f}k"
    return f"{sign}{symbol}{a:,.0f}"


_ACRONYMS = {"ROI"}


def priority_label(priority: str) -> str:
    """'WIN BACK' -> 'Win back', 'ASSESS ROI' -> 'Assess ROI' (``str.title`` mangles acronyms)."""
    words = [w if w in _ACRONYMS else w.lower() for w in priority.split()]
    if words and words[0] not in _ACRONYMS:
        words[0] = words[0].capitalize()
    return " ".join(words)


def fmt_count(value: float) -> str:
    """Whole number with thousands separators; one decimal if not integral."""
    v = float(value)
    return f"{v:,.0f}" if abs(v - round(v)) < 0.05 else f"{v:,.1f}"


# ---------------------------------------------------------------------------
# Segment summary
# ---------------------------------------------------------------------------
def present_segments(rfm: pd.DataFrame) -> list[str]:
    """Segments that have at least one customer, in business order."""
    have = set(rfm["Segment"].unique())
    return [s for s in SEGMENT_ORDER if s in have]


def summarize_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Per-segment KPIs computed from the (possibly filtered) customer table.

    Column names match ``segment_summary.csv`` (Customers, Avg_Recency,
    Avg_Frequency, Avg_Monetary, Total_Revenue, Revenue_Pct) plus extras used
    by the dashboard.
    """
    if rfm.empty:
        return pd.DataFrame(columns=[
            "Segment", "Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary",
            "Total_Revenue", "Revenue_Pct", "Customer_Pct", "Revenue_per_Customer",
            "Avg_Order_Value", "Median_Recency", "Median_Frequency", "Median_Monetary",
        ])
    seg = (
        rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
            Total_Orders=("Frequency", "sum"),
            Median_Recency=("Recency", "median"),
            Median_Frequency=("Frequency", "median"),
            Median_Monetary=("Monetary", "median"),
        )
        .reindex(present_segments(rfm))
        .reset_index()
    )
    seg["Revenue_Pct"] = seg["Total_Revenue"] / seg["Total_Revenue"].sum() * 100
    seg["Customer_Pct"] = seg["Customers"] / seg["Customers"].sum() * 100
    seg["Revenue_per_Customer"] = seg["Total_Revenue"] / seg["Customers"]
    seg["Avg_Order_Value"] = seg["Total_Revenue"] / seg["Total_Orders"]
    return seg.drop(columns="Total_Orders")


def overall_metrics(rfm: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for a set of customers."""
    orders = float(rfm["Frequency"].sum())
    revenue = float(rfm["Monetary"].sum())
    return {
        "customers": float(len(rfm)),
        "revenue": revenue,
        "avg_customer_value": float(rfm["Monetary"].mean()),
        "avg_order_value": revenue / orders if orders else 0.0,
        "avg_orders": float(rfm["Frequency"].mean()),
        "median_recency": float(rfm["Recency"].median()),
        "median_frequency": float(rfm["Frequency"].median()),
        "median_monetary": float(rfm["Monetary"].median()),
    }


# ---------------------------------------------------------------------------
# Revenue concentration
# ---------------------------------------------------------------------------
def lorenz_curve(monetary: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative revenue share when customers are ranked from biggest spender down.

    Returns (customer_share, revenue_share), both in [0, 1], starting at (0, 0).
    """
    values = np.sort(np.asarray(monetary, dtype=float))[::-1]
    total = values.sum()
    if values.size == 0 or total <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.concatenate([[0.0], np.arange(1, values.size + 1) / values.size])
    y = np.concatenate([[0.0], np.cumsum(values) / total])
    return x, y


def top_share(monetary: pd.Series, fraction: float) -> float:
    """Share of total revenue (0-1) earned by the top ``fraction`` of customers."""
    values = np.sort(np.asarray(monetary, dtype=float))[::-1]
    total = values.sum()
    if values.size == 0 or total <= 0:
        return 0.0
    n = min(values.size, max(1, round(values.size * fraction)))
    return float(values[:n].sum() / total)


# ---------------------------------------------------------------------------
# Segment profile (what each segment looks like)
# ---------------------------------------------------------------------------
def segment_profile(rfm: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Median R/F/M per segment plus a signed 'better than typical' score.

    The score is ``log2(segment median / typical median)`` for Frequency and
    Monetary and the inverse for Recency (lower days are better), so positive
    always means "better than the typical customer". It is clipped to +/-3
    (8x better / worse) for colouring.
    """
    ref = rfm if ref is None else ref
    typical = ref[FEATURES].median()
    med = rfm.groupby("Segment")[FEATURES].median().reindex(present_segments(rfm))
    score = pd.DataFrame(index=med.index)
    score["Recency"] = np.log2(typical["Recency"] / med["Recency"].clip(lower=1))
    score["Frequency"] = np.log2(med["Frequency"] / max(typical["Frequency"], 1))
    score["Monetary"] = np.log2(med["Monetary"] / max(typical["Monetary"], 1))
    return med.join(score.clip(-3, 3), rsuffix="_score")


def band_matrix(
    rfm: pd.DataFrame,
    recency_bins: list[float],
    recency_labels: list[str],
    monetary_bins: list[float],
    monetary_labels: list[str],
    metric: str = "customers",
) -> pd.DataFrame:
    """Recency x spend grid (rows = spend bands, columns = recency bands).

    ``metric`` is ``"customers"`` (counts) or ``"revenue"`` (percent of the
    revenue in view). The last band in each axis is open-ended.
    """
    if len(recency_bins) != len(recency_labels) or len(monetary_bins) != len(monetary_labels):
        raise ValueError("Heatmap bins and labels must have the same length.")
    frame = pd.DataFrame({
        "r": pd.cut(rfm["Recency"], bins=[*recency_bins, np.inf], labels=recency_labels,
                    right=True, include_lowest=True),
        "m": pd.cut(rfm["Monetary"], bins=[*monetary_bins, np.inf], labels=monetary_labels,
                    right=True, include_lowest=True),
        "value": rfm["Monetary"] if metric == "revenue" else 1.0,
    })
    grid = (
        frame.groupby(["m", "r"], observed=False)["value"].sum()
        .unstack(fill_value=0)
        .reindex(index=monetary_labels, columns=recency_labels, fill_value=0)
    )
    if metric == "revenue":
        total = grid.to_numpy().sum()
        grid = grid / total * 100 if total else grid
    return grid


# ---------------------------------------------------------------------------
# Auto-generated insights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Insight:
    tag: str
    headline: str
    detail: str
    color: str


def generate_insights(
    rfm: pd.DataFrame,
    summary: pd.DataFrame,
    symbol: str,
    params: dict[str, Any],
) -> list[Insight]:
    """Plain-English takeaways computed from the customers currently in view."""
    out: list[Insight] = []
    n = len(rfm)
    if n == 0:
        return out

    # 1. Concentration
    if n >= 10:
        t20, t1 = top_share(rfm["Monetary"], 0.20), top_share(rfm["Monetary"], 0.01)
        n1 = max(1, round(n * 0.01))
        out.append(Insight(
            "Revenue concentration",
            f"Top 20% of customers drive {t20 * 100:.0f}% of revenue",
            f"The top 1% alone ({n1:,} customers) account for {t1 * 100:.0f}%.",
            ACCENT,
        ))

    # 2. Star segment (highest value per customer)
    star = summary.sort_values("Revenue_per_Customer", ascending=False).iloc[0] if len(summary) else None
    avg = rfm["Monetary"].mean()
    mult = star["Revenue_per_Customer"] / avg if star is not None and avg else 0
    if len(summary) > 1 and mult >= 1.5:  # only worth a card if the segment stands out
        out.append(Insight(
            "Star segment",
            f"{star['Segment']}: {star['Customer_Pct']:.1f}% of customers, "
            f"{star['Revenue_Pct']:.0f}% of revenue",
            f"Each is worth {mult:.0f}x the average customer "
            f"({fmt_money(star['Revenue_per_Customer'], symbol)} vs {fmt_money(avg, symbol)}).",
            SEGMENT_COLORS.get(star["Segment"], ACCENT),
        ))

    # 3. Win-back pool
    min_orders = int(params.get("at_risk_min_orders", 3))
    at_risk = rfm[rfm["Segment"] == "At Risk"]
    if not at_risk.empty:
        pool = at_risk[at_risk["Frequency"] >= min_orders]
        if not pool.empty:
            out.append(Insight(
                "Best win-back pool",
                f"{len(pool):,} At Risk customers with {min_orders}+ orders "
                f"hold {fmt_money(pool['Monetary'].sum(), symbol)}",
                f"That is {len(pool) / len(at_risk) * 100:.0f}% of the At Risk segment "
                f"but {pool['Monetary'].sum() / at_risk['Monetary'].sum() * 100:.0f}% of its spend.",
                AMBER,
            ))

    # 4. One-time buyers
    one = (rfm["Frequency"] == 1).mean() * 100
    out.append(Insight(
        "One-time buyers",
        f"{one:.0f}% of customers have ordered only once",
        "A second order is the cheapest growth lever: these customers already know you.",
        SEGMENT_COLORS["Potential Loyalists"],
    ))
    return out


# ---------------------------------------------------------------------------
# Action playbook: sized target lists
# ---------------------------------------------------------------------------
TARGET_DESCRIPTIONS: dict[str, str] = {
    "Champions": "No purchase in the last {champion_watch_days} days (VIPs slipping away)",
    "Loyal Customers": "Top {loyal_top_share:.0%} by RFM score, then spend (Champion candidates)",
    "Potential Loyalists": "Bought within {potential_recent_days} days, with order count between "
                           "this segment's median and the Loyal tier's (closest to levelling up)",
    "At Risk": "{at_risk_min_orders}+ orders (proven buyers gone quiet)",
    "Lost / Churned": "Top {lost_top_share:.0%} by historical spend (only reactivate the best)",
}


def _top_share_index(seg: pd.DataFrame, share: float, sort_cols: list[str]) -> pd.Index:
    n = max(1, math.ceil(len(seg) * share))
    return seg.sort_values(sort_cols, ascending=False).head(n).index


def target_description(segment: str, params: dict[str, Any]) -> str:
    """Human-readable definition of a segment's priority target group."""
    return TARGET_DESCRIPTIONS[segment].format(**params)


def target_mask(
    seg: pd.DataFrame,
    segment: str,
    ref: pd.DataFrame,
    params: dict[str, Any],
) -> pd.Series:
    """Boolean mask (aligned to ``seg``) of the segment's priority target group.

    ``ref`` is the full, unfiltered customer table; it supplies benchmarks
    (such as the Loyal tier's median order count) that must not move when the
    sidebar filters change.
    """
    if seg.empty:
        return pd.Series(False, index=seg.index)
    if segment == "Champions":
        return seg["Recency"] > params["champion_watch_days"]
    if segment == "Loyal Customers":
        idx = _top_share_index(seg, params["loyal_top_share"], ["RFM_Score", "Monetary"])
        return pd.Series(seg.index.isin(idx), index=seg.index)
    if segment == "Potential Loyalists":
        loyal = ref.loc[ref["Segment"] == "Loyal Customers", "Frequency"]
        bench = loyal.median() if not loyal.empty else ref["Frequency"].median()
        return (
            (seg["Recency"] <= params["potential_recent_days"])
            & (seg["Frequency"] >= seg["Frequency"].median())
            & (seg["Frequency"] < bench)
        )
    if segment == "At Risk":
        return seg["Frequency"] >= params["at_risk_min_orders"]
    if segment == "Lost / Churned":
        idx = _top_share_index(seg, params["lost_top_share"], ["Monetary"])
        return pd.Series(seg.index.isin(idx), index=seg.index)
    raise KeyError(f"Unknown segment: {segment}")


def target_customers(
    rfm: pd.DataFrame, ref: pd.DataFrame, segment: str, params: dict[str, Any],
) -> pd.DataFrame:
    """Customers in the segment's priority target group, biggest spenders first."""
    seg = rfm[rfm["Segment"] == segment]
    picked = seg[target_mask(seg, segment, ref, params)]
    return picked.sort_values("Monetary", ascending=False)


def build_playbook(
    rfm: pd.DataFrame, ref: pd.DataFrame, params: dict[str, Any],
) -> pd.DataFrame:
    """One row per segment in view: priority, target group and its size / value."""
    rows = []
    for segment in present_segments(rfm):
        seg = rfm[rfm["Segment"] == segment]
        tgt = target_customers(rfm, ref, segment, params)
        orders = float(tgt["Frequency"].sum())
        spend = float(tgt["Monetary"].sum())
        rows.append({
            "Segment": segment,
            "Priority": RECOMMENDATIONS[segment]["priority"],
            "Target group": target_description(segment, params),
            "Segment customers": len(seg),
            "Targeted customers": len(tgt),
            "Share of segment": len(tgt) / len(seg) if len(seg) else 0.0,
            "Targeted spend": spend,
            "Share of segment spend": spend / seg["Monetary"].sum() if len(seg) else 0.0,
            "Avg order value": spend / orders if orders else 0.0,
        })
    return pd.DataFrame(rows)


def campaign_economics(
    targeted: int,
    avg_order_value: float,
    response_rate: float,
    cost_per_contact: float,
    gross_margin: float,
    discount: float,
) -> dict[str, float | None]:
    """Expected campaign outcome under simple, explicit assumptions.

    Assumes each responder places exactly one order at the target group's
    historical average order value, discounted by ``discount``; profit per
    responder is ``AOV * (gross_margin - discount)``. Repeat purchases are
    ignored, which makes the estimate conservative.

    Returns a dict with responders, revenue, gross_profit, cost, net_profit,
    roi (net / cost, ``None`` when cost is zero) and breakeven_response
    (``None`` when a responder is not profitable at all).
    """
    responders = targeted * response_rate
    profit_per_responder = avg_order_value * (gross_margin - discount)
    revenue = responders * avg_order_value * (1 - discount)
    gross_profit = responders * profit_per_responder
    cost = targeted * cost_per_contact
    net = gross_profit - cost
    return {
        "responders": responders,
        "revenue": revenue,
        "gross_profit": gross_profit,
        "cost": cost,
        "net_profit": net,
        "roi": net / cost if cost > 0 else None,
        "breakeven_response": cost_per_contact / profit_per_responder if profit_per_responder > 0 else None,
    }


# ---------------------------------------------------------------------------
# Customer 360
# ---------------------------------------------------------------------------
def customer_percentiles(row: pd.Series, ref: pd.DataFrame) -> dict[str, float]:
    """Share of all customers this customer beats on each metric (0-100, higher = better).

    Recency is inverted: a customer who bought more recently than 90% of
    others scores 90. Ties are not counted as "beaten".
    """
    return {
        "Recency": float((ref["Recency"] > row["Recency"]).mean() * 100),
        "Frequency": float((ref["Frequency"] < row["Frequency"]).mean() * 100),
        "Monetary": float((ref["Monetary"] < row["Monetary"]).mean() * 100),
    }
