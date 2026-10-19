"""Plotly figure builders for the dashboard.

Design rules applied to every chart:
    * direct labels instead of legends wherever possible
    * log scales for heavily skewed metrics (orders, spend)
    * a short axis title that says which direction is "better"
    * colour carries one meaning only (segment identity, or better/worse)

Builders are pure (DataFrame in, Figure out) so they can be tested headlessly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import band_matrix, fmt_count, fmt_money, lorenz_curve, present_segments, top_share
from constants import (
    ACCENT,
    AMBER,
    BAD_TINT,
    GOOD_TINT,
    GREEN,
    MID_TINT,
    MUTED,
    NAVY,
    NAVY_BORDER,
    NAVY_CARD,
    NAVY_RAISED,
    RED,
    SEGMENT_COLORS,
    SEGMENT_ORDER,
    SUBTLE,
    WHITE,
)

PAIR_HEIGHT = 360  # charts shown side by side share one height so their panels align
FONT = "DM Sans, system-ui, -apple-system, Segoe UI, sans-serif"
GRID = "rgba(141,162,190,0.14)"


def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert a hex colour to an rgba() CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(height: int = 300, **overrides: Any) -> dict[str, Any]:
    """Shared transparent layout; keyword overrides replace top-level keys."""
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT, "color": WHITE, "size": 12},
        "height": height,
        "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
        "hoverlabel": {"bgcolor": NAVY_RAISED, "bordercolor": NAVY_BORDER,
                       "font": {"family": FONT, "color": WHITE, "size": 12}},
    }
    layout.update(overrides)
    return layout


def _axis(**kw: Any) -> dict[str, Any]:
    axis: dict[str, Any] = {
        "gridcolor": GRID, "zeroline": False, "linecolor": NAVY_BORDER,
        "tickfont": {"color": MUTED, "size": 11}, "title_font": {"color": MUTED, "size": 11},
    }
    axis.update(kw)
    return axis


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
def share_bars(summary: pd.DataFrame) -> go.Figure:
    """Share of customers (grey) next to share of revenue (segment colour)."""
    segs = list(summary["Segment"])
    cust = summary["Customer_Pct"].to_numpy()
    rev = summary["Revenue_Pct"].to_numpy()
    fig = go.Figure()
    fig.add_bar(
        y=segs, x=cust, orientation="h", name="Share of customers",
        marker={"color": SUBTLE}, text=[f"{v:.1f}%" for v in cust], textposition="outside",
        textfont={"color": MUTED, "size": 11}, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of customers<extra></extra>",
    )
    fig.add_bar(
        y=segs, x=rev, orientation="h", name="Share of revenue",
        marker={"color": [SEGMENT_COLORS.get(s, MUTED) for s in segs]},
        text=[f"{v:.1f}%" for v in rev], textposition="outside",
        textfont={"color": WHITE, "size": 12}, cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of revenue<extra></extra>",
    )
    top = max(float(cust.max()), float(rev.max()))
    fig.update_layout(**base_layout(
        height=PAIR_HEIGHT, barmode="group", bargap=0.35, bargroupgap=0.08,
        showlegend=False,
        xaxis=_axis(showticklabels=False, showgrid=False, range=[0, top * 1.22]),
        yaxis=_axis(categoryorder="array", categoryarray=segs, autorange="reversed",
                    showgrid=False, tickfont={"color": WHITE, "size": 12}),
        margin={"l": 8, "r": 24, "t": 8, "b": 8},
    ))
    return fig


def pareto_figure(monetary: pd.Series, symbol: str = "£") -> go.Figure:
    """Cumulative revenue vs customers, with the 1% / 10% / 20% / 50% points labelled."""
    x, y = lorenz_curve(monetary)
    n = len(monetary)
    idx = np.unique(np.linspace(0, len(x) - 1, min(len(x), 400)).astype(int))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines", name="If every customer spent the same",
        line={"color": SUBTLE, "dash": "dot", "width": 1.5}, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x[idx] * 100, y=y[idx] * 100, mode="lines", name="Actual",
        line={"color": ACCENT, "width": 3}, fill="tonexty", fillcolor=hex_to_rgba(ACCENT, 0.16),
        hovertemplate="Top %{x:.0f}% of customers<br>= %{y:.0f}% of revenue<extra></extra>",
    ))
    for pct, pos in ((1, "bottom right"), (10, "bottom right"), (20, "bottom right"), (50, "bottom right")):
        if n * pct / 100 < 1:
            continue
        share = top_share(monetary, pct / 100) * 100
        fig.add_trace(go.Scatter(
            x=[pct], y=[share], mode="markers+text", showlegend=False,
            marker={"color": WHITE, "size": 9, "line": {"color": ACCENT, "width": 2}},
            text=[f"Top {pct}% → {share:.0f}%"], textposition=pos,
            textfont={"color": WHITE, "size": 11}, hoverinfo="skip",
        ))
    fig.update_layout(**base_layout(
        height=PAIR_HEIGHT, showlegend=False,
        xaxis=_axis(title="Customers, ranked from biggest spender (%)", range=[0, 100],
                    ticksuffix="%", dtick=20),
        yaxis=_axis(title="Cumulative share of revenue", range=[0, 102], ticksuffix="%", dtick=20),
        margin={"l": 8, "r": 16, "t": 8, "b": 8},
    ))
    return fig


_DIVERGING = [[0.0, BAD_TINT], [0.5, MID_TINT], [1.0, GOOD_TINT]]


def profile_heatmap(profile: pd.DataFrame, symbol: str = "£") -> go.Figure:
    """Median R/F/M per segment, coloured by better / worse than the typical customer."""
    segs = list(profile.index)
    cols = ["Recency", "Frequency", "Monetary"]
    z = profile[[f"{c}_score" for c in cols]].to_numpy()
    text = []
    for seg in segs:
        r, f, m = (profile.loc[seg, c] for c in cols)
        text.append([f"{r:.0f} days", f"{fmt_count(f)} {'order' if round(f) == 1 else 'orders'}",
                     fmt_money(m, symbol)])
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=segs, text=text, texttemplate="%{text}",
        textfont={"color": WHITE, "size": 13}, colorscale=_DIVERGING, zmin=-3, zmax=3,
        showscale=False, xgap=3, ygap=3,
        hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(**base_layout(
        height=PAIR_HEIGHT,
        xaxis=_axis(side="top", showgrid=False, ticktext=[
            "Recency<br><span style='font-size:10px;color:#8DA2BE'>days since last order</span>",
            "Frequency<br><span style='font-size:10px;color:#8DA2BE'>orders placed</span>",
            "Spend<br><span style='font-size:10px;color:#8DA2BE'>lifetime total</span>",
        ], tickvals=cols, tickfont={"color": WHITE, "size": 12}),
        yaxis=_axis(autorange="reversed", showgrid=False, tickfont={"color": WHITE, "size": 12}),
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
    ))
    return fig


def band_heatmap(
    rfm: pd.DataFrame,
    rec_bins: list[float], rec_labels: list[str],
    mon_bins: list[float], mon_labels: list[str],
    metric: str, symbol: str = "£",
) -> go.Figure:
    """Recency x spend grid. ``metric`` is 'customers' or 'revenue'."""
    grid = band_matrix(rfm, rec_bins, rec_labels, mon_bins, mon_labels, metric)
    pretty_y = [f"{symbol}{lbl}" for lbl in mon_labels]
    if metric == "revenue":
        text = [[f"{v:.1f}%" if v >= 0.05 else "" for v in row] for row in grid.to_numpy()]
        hover = "%{y} spend, %{x} days<br>%{z:.1f}% of revenue<extra></extra>"
    else:
        text = [[f"{int(v):,}" if v else "" for v in row] for row in grid.to_numpy()]
        hover = "%{y} spend, %{x} days<br>%{z:,} customers<extra></extra>"
    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=rec_labels, y=pretty_y, text=text, texttemplate="%{text}",
        textfont={"color": WHITE, "size": 12}, showscale=False, xgap=3, ygap=3,
        colorscale=[[0.0, NAVY_CARD], [0.02, "#17284A"], [1.0, ACCENT]],
        hovertemplate=hover,
    ))
    fig.update_layout(**base_layout(
        height=PAIR_HEIGHT,
        xaxis=_axis(title="Days since last purchase  (further right = less recent)",
                    showgrid=False),
        yaxis=_axis(title=f"Lifetime spend ({symbol})", autorange="reversed", showgrid=False),
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
    ))
    return fig


# ---------------------------------------------------------------------------
# Segment deep dive
# ---------------------------------------------------------------------------
def _decade_ticks(lo: float, hi: float, symbol: str, money: bool) -> tuple[list[float], list[str]]:
    vals, labels = [], []
    for p in range(int(np.floor(np.log10(max(lo, 1e-9)))), int(np.ceil(np.log10(max(hi, 1)))) + 1):
        v = 10 ** p
        if lo / 1.5 <= v <= hi * 1.5:
            vals.append(float(np.log10(v)))
            labels.append(fmt_money(v, symbol) if money else f"{v:,.0f}")
    return vals, labels


def distribution_figure(
    segment_values: pd.Series, all_values: pd.Series, metric: str, color: str, symbol: str = "£",
) -> go.Figure:
    """Share of customers per value band: this segment (bars) vs everyone (dashed line).

    Frequency and Monetary use log-spaced bands because both are heavily
    right-skewed; Recency uses linear bands.
    """
    all_v = np.asarray(all_values, dtype=float)
    seg_v = np.asarray(segment_values, dtype=float)
    log = metric in ("Frequency", "Monetary")
    lo, hi = float(all_v.min()), float(all_v.max())
    if log:
        lo = max(lo, 1e-2)
        edges = np.logspace(np.log10(lo), np.log10(hi * 1.001), 26)
        mid = (np.log10(edges[:-1]) + np.log10(edges[1:])) / 2
        width = np.diff(np.log10(edges))
        seg_med, all_med = np.log10(max(np.median(seg_v), lo)), np.log10(max(np.median(all_v), lo))
    else:
        edges = np.linspace(lo, hi + 1, 26)
        mid = (edges[:-1] + edges[1:]) / 2
        width = np.diff(edges)
        seg_med, all_med = float(np.median(seg_v)), float(np.median(all_v))
    seg_pct = np.histogram(seg_v, bins=edges)[0] / max(len(seg_v), 1) * 100
    all_pct = np.histogram(all_v, bins=edges)[0] / max(len(all_v), 1) * 100

    fig = go.Figure()
    fig.add_bar(x=mid, y=seg_pct, width=width * 0.92, marker={"color": hex_to_rgba(color, 0.85)},
                name="This segment", hovertemplate="%{y:.1f}% of this segment<extra></extra>")
    fig.add_trace(go.Scatter(
        x=mid, y=all_pct, mode="lines", name="All customers",
        line={"color": MUTED, "width": 2, "dash": "dot"},
        hovertemplate="%{y:.1f}% of all customers<extra></extra>",
    ))
    fig.add_vline(x=seg_med, line={"color": color, "width": 2})
    fig.add_vline(x=all_med, line={"color": MUTED, "width": 1.5, "dash": "dot"})

    xaxis = _axis(showgrid=False)
    if log:
        tickvals, ticktext = _decade_ticks(lo, hi, symbol, money=(metric == "Monetary"))
        xaxis.update(tickvals=tickvals, ticktext=ticktext)
    fig.update_layout(**base_layout(
        height=240, showlegend=False, bargap=0, xaxis=xaxis,
        yaxis=_axis(ticksuffix="%", rangemode="tozero"),
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
    ))
    return fig


# ---------------------------------------------------------------------------
# Customer explorer
# ---------------------------------------------------------------------------
def customer_map(rfm: pd.DataFrame, highlight: pd.Series | None = None, symbol: str = "£") -> go.Figure:
    """Every customer as a dot: recency (x) vs lifetime spend (y, log). Top-left = best."""
    fig = go.Figure()
    for seg in present_segments(rfm):
        sub = rfm[rfm["Segment"] == seg]
        fig.add_trace(go.Scattergl(
            x=sub["Recency"], y=sub["Monetary"], mode="markers", name=seg,
            marker={"color": SEGMENT_COLORS.get(seg, MUTED), "size": 6, "opacity": 0.6},
            customdata=np.stack([sub["CustomerID"], sub["Frequency"]], axis=-1),
            hovertemplate=(f"<b>{seg}</b><br>Customer %{{customdata[0]}}<br>"
                           f"%{{x}} days since last order<br>{symbol}%{{y:,.0f}} spent<br>"
                           "%{customdata[1]} orders<extra></extra>"),
        ))
    if highlight is not None:
        fig.add_trace(go.Scatter(
            x=[highlight["Recency"]], y=[highlight["Monetary"]], mode="markers+text",
            name="Selected customer", showlegend=False,
            marker={"symbol": "star", "size": 18, "color": WHITE, "line": {"color": ACCENT, "width": 2}},
            text=[f"  {int(highlight['CustomerID'])}"], textposition="middle right",
            textfont={"color": WHITE, "size": 12}, hoverinfo="skip",
        ))
    fig.update_layout(**base_layout(
        height=420,
        xaxis=_axis(title="Days since last order  (further right = less recent)"),
        yaxis=_axis(title=f"Lifetime spend ({symbol}, log scale)", type="log",
                    tickprefix=symbol, tickformat="~s"),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "font": {"color": WHITE, "size": 12}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
    ))
    return fig


def percentile_bars(percentiles: dict[str, float], color: str) -> go.Figure:
    """How a single customer ranks against all customers on each metric (higher = better)."""
    labels = {"Recency": "Recency (bought recently)", "Frequency": "Frequency (orders)",
              "Monetary": "Spend (lifetime)"}
    keys = list(percentiles)
    vals = [percentiles[k] for k in keys]
    fig = go.Figure()
    fig.add_bar(y=[labels[k] for k in keys], x=[100] * len(keys), orientation="h",
                marker={"color": NAVY_RAISED}, hoverinfo="skip", showlegend=False)
    fig.add_bar(
        y=[labels[k] for k in keys], x=vals, orientation="h", marker={"color": color},
        text=[f"better than {v:.0f}%" for v in vals], textposition="inside",
        insidetextanchor="start", textfont={"color": NAVY, "size": 12},
        hoverinfo="skip", showlegend=False,
    )
    fig.update_layout(**base_layout(
        height=190, barmode="overlay", bargap=0.35,
        xaxis=_axis(range=[0, 100], showgrid=False, showticklabels=False),
        yaxis=_axis(autorange="reversed", showgrid=False, tickfont={"color": WHITE, "size": 12}),
        margin={"l": 8, "r": 8, "t": 4, "b": 4},
    ))
    return fig


# ---------------------------------------------------------------------------
# Playbook
# ---------------------------------------------------------------------------
def economics_waterfall(econ: dict[str, float | None], symbol: str = "£") -> go.Figure:
    """Gross profit minus campaign cost = net profit."""
    gross = float(econ["gross_profit"] or 0)
    cost = float(econ["cost"] or 0)
    net = float(econ["net_profit"] or 0)
    fig = go.Figure(go.Waterfall(
        x=["Gross profit<br>from responders", "Campaign<br>cost", "Net profit"],
        y=[gross, -cost, net], measure=["absolute", "relative", "total"],
        text=[fmt_money(gross, symbol), "-" + fmt_money(cost, symbol), fmt_money(net, symbol)],
        textposition="outside", textfont={"color": WHITE, "size": 12},
        increasing={"marker": {"color": GREEN}}, decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": GREEN if net >= 0 else RED}},
        connector={"line": {"color": SUBTLE, "dash": "dot"}}, hoverinfo="skip",
    ))
    fig.update_layout(**base_layout(
        height=300, showlegend=False,
        xaxis=_axis(showgrid=False, tickfont={"color": WHITE, "size": 12}),
        yaxis=_axis(tickprefix=symbol, tickformat="~s"),
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
    ))
    return fig


# ---------------------------------------------------------------------------
# Method & diagnostics
# ---------------------------------------------------------------------------
def k_diagnostics(table: pd.DataFrame, k_final: int) -> go.Figure:
    """Elbow and silhouette side by side, with the chosen K highlighted."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=(
        "Elbow: lower = tighter clusters", "Silhouette: higher = better separated"))
    fig.add_trace(go.Scatter(x=table["K"], y=table["Inertia"], mode="lines+markers",
                             line={"color": ACCENT, "width": 3}, marker={"size": 8},
                             hovertemplate="K=%{x}<br>Inertia %{y:,.0f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=table["K"], y=table["Silhouette"], mode="lines+markers",
                             line={"color": GREEN, "width": 3}, marker={"size": 8},
                             hovertemplate="K=%{x}<br>Silhouette %{y:.3f}<extra></extra>"), 1, 2)
    best_k = int(table.loc[table["Silhouette"].idxmax(), "K"])
    for col in (1, 2):
        fig.add_vline(x=k_final, line={"color": AMBER, "dash": "dash", "width": 2}, row=1, col=col)
    if (table["K"] == k_final).any():
        fig.add_annotation(
            x=k_final, y=float(table.loc[table["K"] == k_final, "Silhouette"].iloc[0]),
            xref="x2", yref="y2", text=f"chosen K={k_final}", showarrow=True, arrowcolor=AMBER,
            font={"color": AMBER, "size": 12}, ax=40, ay=-40,
        )
    if best_k != k_final:
        fig.add_annotation(
            x=best_k, y=float(table["Silhouette"].max()), xref="x2", yref="y2",
            text=f"best separation: K={best_k}", showarrow=True, arrowcolor=MUTED,
            font={"color": MUTED, "size": 11}, ax=75, ay=-5,
        )
    fig.update_xaxes(**_axis(title="Number of clusters (K)", dtick=1, showgrid=False))
    fig.update_yaxes(**_axis())
    fig.update_layout(**base_layout(height=330, showlegend=False,
                                    margin={"l": 8, "r": 8, "t": 36, "b": 8}))
    for ann in fig.layout.annotations[:2]:
        ann.update(font={"color": WHITE, "size": 12})
    return fig


def score_distribution(rfm: pd.DataFrame) -> go.Figure:
    """RFM score (3-15) histogram, stacked by segment."""
    scores = list(range(int(rfm["RFM_Score"].min()), int(rfm["RFM_Score"].max()) + 1))
    fig = go.Figure()
    for seg in SEGMENT_ORDER:
        sub = rfm[rfm["Segment"] == seg]
        if sub.empty:
            continue
        counts = sub["RFM_Score"].value_counts().reindex(scores, fill_value=0)
        fig.add_bar(x=scores, y=counts.values, name=seg, marker={"color": SEGMENT_COLORS[seg]},
                    hovertemplate=f"<b>{seg}</b><br>Score %{{x}}: %{{y:,}} customers<extra></extra>")
    fig.update_layout(**base_layout(
        height=300, barmode="stack", bargap=0.12,
        xaxis=_axis(title="RFM score (sum of three 1-5 scores)", dtick=1, showgrid=False),
        yaxis=_axis(title="Customers"),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0,
                "font": {"color": WHITE, "size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
    ))
    return fig
