"""
Customer Segmentation Dashboard — Streamlit application.

Interactive dark-themed UI for exploring the RFM / K-Means customer segments
produced by :mod:`rfm_analysis`. Every number on screen is derived from the
customers currently in view (sidebar filters), via :mod:`analytics`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import pandas as pd
import streamlit as st

import ui
from analytics import (
    build_playbook,
    campaign_economics,
    customer_percentiles,
    fmt_count,
    fmt_money,
    generate_insights,
    overall_metrics,
    present_segments,
    priority_label,
    segment_profile,
    summarize_segments,
    target_customers,
    top_share,
)
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
from constants import (
    ACCENT,
    BAD_TINT,
    GOOD_TINT,
    GREEN,
    MUTED,
    RECOMMENDATIONS,
    RED,
    SEGMENT_COLORS,
    SEGMENT_DESCRIPTIONS,
    SEGMENT_ICONS,
    SEGMENT_ORDER,
    SUBTLE,
    WHITE,
)

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.inject_css()

# ── Configuration ─────────────────────────────────────────────────────────────
try:
    CFG: dict[str, Any] = load_config()
except FileNotFoundError:
    st.error("config.yaml not found in the project root — restore it and reload the app.")
    st.stop()

CURRENCY: str = CFG.get("display", {}).get("currency_symbol", "£")
REC_BINS: list[int] = CFG["heatmap"]["recency_bins"]
REC_LABELS: list[str] = CFG["heatmap"]["recency_labels"]
MON_BINS: list[int] = CFG["heatmap"]["monetary_bins"]
MON_LABELS: list[str] = CFG["heatmap"]["monetary_labels"]
SPEND_COL = f"Spend ({CURRENCY})"

PLAYBOOK_PARAMS: dict[str, Any] = {
    "champion_watch_days": 45, "loyal_top_share": 0.20, "potential_recent_days": 30,
    "at_risk_min_orders": 3, "lost_top_share": 0.25,
    **CFG.get("playbook", {}),
}
ROI_DEFAULTS: dict[str, float] = {
    "cost_per_contact": 0.50, "gross_margin": 0.30, "discount": 0.10, "response_rate": 0.05,
    **CFG.get("roi_defaults", {}),
}

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, "data")
RFM_PATH = os.path.join(DATA_DIR, "rfm_scored.csv")
META_PATH = os.path.join(DATA_DIR, "run_metadata.json")
REQUIRED_COLUMNS = {"CustomerID", "Recency", "Frequency", "Monetary",
                    "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"}


def _data_mtime() -> float:
    """Latest modification time of the pipeline outputs (cache key)."""
    paths = (RFM_PATH, META_PATH, os.path.join(BASE_DIR, "config.yaml"))
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)), default=0.0)


@st.cache_data(ttl=86400)
def _load_rfm(mtime: float) -> pd.DataFrame:
    return pd.read_csv(RFM_PATH)


@st.cache_data(ttl=86400)
def _load_meta(mtime: float) -> dict[str, Any]:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=86400, show_spinner="Computing cluster diagnostics...")
def _load_diagnostics(mtime: float) -> pd.DataFrame:
    return diagnostics_table(_load_rfm(mtime), CFG)


@st.cache_data(max_entries=64)
def to_csv(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Rename pipeline columns to the display names used by every table and CSV export."""
    out = df.copy()
    if "Monetary" in out:
        out["Monetary"] = out["Monetary"].round(2)
    return out.rename(columns={
        "CustomerID": "Customer ID", "Recency": "Days since last order", "Frequency": "Orders",
        "Monetary": SPEND_COL, "R_Score": "R", "F_Score": "F", "M_Score": "M",
        "RFM_Score": "RFM score", "Priority_Group": "Priority group",
    })


if not os.path.exists(RFM_PATH):
    st.error("data/rfm_scored.csv not found. Run `python rfm_analysis.py` to generate it.")
    st.stop()

MTIME = _data_mtime()
rfm_full = _load_rfm(MTIME)
META = _load_meta(MTIME)
if not REQUIRED_COLUMNS <= set(rfm_full.columns):
    st.error(f"data/rfm_scored.csv is missing columns: {sorted(REQUIRED_COLUMNS - set(rfm_full.columns))}. "
             "Re-run `python rfm_analysis.py`.")
    st.stop()

FULL = overall_metrics(rfm_full)

# ── Sidebar filter state ──────────────────────────────────────────────────────
SCORE_MIN, SCORE_MAX = int(rfm_full["RFM_Score"].min()), int(rfm_full["RFM_Score"].max())
REC_MIN, REC_MAX = int(rfm_full["Recency"].min()), int(rfm_full["Recency"].max())
SPEND_MAX = int(math.ceil(rfm_full["Monetary"].max()))  # ceil: int() would drop the top spender
_ladder = sorted({m * 10 ** e for e in range(0, 9) for m in (1, 2, 5)})
SPEND_OPTIONS: list[int] = [0] + [v for v in _ladder if 0 < v < SPEND_MAX] + [SPEND_MAX]

DEFAULT_FILTERS: dict[str, Any] = {
    "f_segments": list(SEGMENT_ORDER),
    "f_score": (SCORE_MIN, SCORE_MAX),
    "f_recency": (REC_MIN, REC_MAX),
    "f_spend": (SPEND_OPTIONS[0], SPEND_OPTIONS[-1]),
}
PRESETS: dict[str, list[str]] = {
    "All customers": list(SEGMENT_ORDER),
    "Revenue drivers": ["Champions", "Loyal Customers"],
    "Growth pool": ["Potential Loyalists"],
    "Needs attention": ["At Risk", "Lost / Churned"],
}


def _apply_preset() -> None:
    choice = st.session_state["quick_view"]
    if choice in PRESETS:
        for key, default in DEFAULT_FILTERS.items():
            st.session_state[key] = default
        st.session_state["f_segments"] = list(PRESETS[choice])


def _mark_custom() -> None:
    st.session_state["quick_view"] = "Custom"


def _reset_filters() -> None:
    for key, default in DEFAULT_FILTERS.items():
        st.session_state[key] = default
    st.session_state["quick_view"] = "All customers"


with st.sidebar:
    ui.render("""
    <div class="side-brand">◈ Segmentation</div>
    <div class="side-sub">Customer intelligence dashboard</div>""")

    ui.render('<div class="side-label">Quick view</div>')
    st.selectbox("Quick view", ["Custom", *PRESETS], index=1, key="quick_view",
                 on_change=_apply_preset, label_visibility="collapsed")

    ui.render('<div class="side-label">Segments</div>')
    st.multiselect("Segments", options=SEGMENT_ORDER, default=DEFAULT_FILTERS["f_segments"], key="f_segments",
                   on_change=_mark_custom, label_visibility="collapsed")

    ui.render('<div class="side-label">RFM score (3 = weakest, 15 = strongest)</div>')
    st.slider("RFM score", SCORE_MIN, SCORE_MAX, value=DEFAULT_FILTERS["f_score"], key="f_score",
              on_change=_mark_custom, label_visibility="collapsed")

    ui.render('<div class="side-label">Days since last purchase</div>')
    st.slider("Days since last purchase", REC_MIN, REC_MAX, value=DEFAULT_FILTERS["f_recency"],
              key="f_recency",
              on_change=_mark_custom, label_visibility="collapsed")

    ui.render(f'<div class="side-label">Lifetime spend ({ui.esc(CURRENCY)})</div>')
    st.select_slider("Lifetime spend", options=SPEND_OPTIONS, value=DEFAULT_FILTERS["f_spend"],
                     key="f_spend",
                     format_func=lambda v: fmt_money(v, CURRENCY),
                     on_change=_mark_custom, label_visibility="collapsed")

    st.button("Reset filters", on_click=_reset_filters, key="reset_filters")

# ── Apply filters ─────────────────────────────────────────────────────────────
_segs = st.session_state["f_segments"]
_score = st.session_state["f_score"]
_rec = st.session_state["f_recency"]
_spend = st.session_state["f_spend"]

rfm = rfm_full[
    rfm_full["Segment"].isin(_segs)
    & rfm_full["RFM_Score"].between(*_score)
    & rfm_full["Recency"].between(*_rec)
    & rfm_full["Monetary"].between(*_spend)
].copy()

if rfm.empty:
    st.warning("No customers match the current filters. Try broadening your selection.")
    st.stop()

seg = summarize_segments(rfm)
VIEW = overall_metrics(rfm)
is_filtered = len(rfm) != len(rfm_full)

with st.sidebar:
    ui.render('<div class="side-label">Segments in view</div>')
    counts = rfm["Segment"].value_counts()
    ui.render("".join(
        f'<div class="legend-row"><div class="dot" style="background:{color};"></div>'
        f'<div class="name">{ui.esc(name)}</div><div class="n">{int(counts.get(name, 0)):,}</div></div>'
        for name, color in SEGMENT_COLORS.items()
    ))

# ── Page header ───────────────────────────────────────────────────────────────
asof = META.get("snapshot_date")
ui.page_header(
    "Customer Segmentation",
    "RFM scoring + K-Means clustering · who your customers are, what they are worth, and what to do next",
    f"Online Retail · as of {asof}" if asof else "Online Retail dataset",
)

if is_filtered:
    rev_share = VIEW["revenue"] / FULL["revenue"] * 100 if FULL["revenue"] else 0.0
    ui.render(f'<div class="filter-chip">Filtered view · <b>{len(rfm):,}</b> of {len(rfm_full):,} customers '
              f'· {rev_share:.0f}% of revenue</div>')

ui.spacer("lg")

# ── KPI cards ─────────────────────────────────────────────────────────────────
kpis = [
    ("Customers", fmt_count(VIEW["customers"]),
     f"<b>{VIEW['customers'] / FULL['customers'] * 100:.0f}%</b> of all {FULL['customers']:,.0f}"),
    ("Revenue", fmt_money(VIEW["revenue"], CURRENCY),
     f"<b>{VIEW['revenue'] / FULL['revenue'] * 100:.0f}%</b> of total revenue"),
    ("Value per customer", fmt_money(VIEW["avg_customer_value"], CURRENCY), "avg. lifetime spend"),
    ("Order value", fmt_money(VIEW["avg_order_value"], CURRENCY), "avg. revenue per order"),
    ("Orders per customer", f"{VIEW['avg_orders']:.1f}", f"median <b>{fmt_count(VIEW['median_frequency'])}</b>"),
    ("Days since last order", f"{VIEW['median_recency']:.0f}", "median across customers"),
]
ui.kpi_row(kpis)

ui.spacer("md")

TABLE_CONFIG = {
    "Customer ID": st.column_config.NumberColumn(format="%d"),
    "Days since last order": st.column_config.NumberColumn(format="%d"),
    "Orders": st.column_config.NumberColumn(format="%d"),
    SPEND_COL: st.column_config.NumberColumn(format="%.0f"),
}

tab_over, tab_deep, tab_play, tab_explore, tab_method = st.tabs([
    "Overview", "Segment Deep Dive", "Action Playbook", "Customer Explorer", "Method & Diagnostics",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_over:
    ui.section("Where the value sits")
    left, right = st.columns(2, gap="large")
    with left:
        if len(seg) > 1:
            star = seg.sort_values("Revenue_per_Customer", ascending=False).iloc[0]
            sub = (f"{ui.key_dot(SUBTLE)}share of customers &nbsp; {ui.key_dot(WHITE)}share of revenue. "
                   f"<b>{ui.esc(star['Segment'])}</b> are {star['Customer_Pct']:.1f}% of customers "
                   f"but {star['Revenue_Pct']:.0f}% of revenue.")
        else:
            sub = "Share of customers vs share of revenue."
        ui.chart_header("Few customers, most of the revenue", sub, lines=3)
        ui.show_plot(share_bars(seg), "ov_share")
    with right:
        t20, t1 = top_share(rfm["Monetary"], 0.20) * 100, top_share(rfm["Monetary"], 0.01) * 100
        ui.chart_header(
            "Revenue is concentrated in group",
            f"The <b>top 20%</b> of customers generate <b>{t20:.0f}%</b> of revenue; "
            f"the top 1% generate <b>{t1:.0f}%</b>. The further the curve bows from the dotted line, "
            "the more unequal.", lines=3)
        ui.show_plot(pareto_figure(rfm["Monetary"], CURRENCY), "ov_pareto")

    ui.section("Who they are")
    left, right = st.columns(2, gap="large")
    with left:
        ui.chart_header(
            "What each segment looks like",
            f"Median customer per segment. {ui.key_dot(GOOD_TINT)}better than the typical customer "
            f"&nbsp; {ui.key_dot(BAD_TINT)}worse.", lines=3)
        ui.show_plot(profile_heatmap(segment_profile(rfm, rfm_full), CURRENCY), "ov_profile")
    with right:
        ui.chart_header(
            "Where customers sit: recency × spend",
            "Top-left = recent big spenders. Bottom-right = inactive, low spend.", lines=3)
        metric_label = st.session_state.get("band_metric", "Customers")
        ui.show_plot(band_heatmap(rfm, REC_BINS, REC_LABELS, MON_BINS, MON_LABELS,
                                  "revenue" if metric_label == "Share of revenue" else "customers", CURRENCY),
                     "ov_band")
        st.radio("Show in grid", ["Customers", "Share of revenue"], horizontal=True, key="band_metric")

    insights = generate_insights(rfm, seg, CURRENCY, PLAYBOOK_PARAMS)
    if insights:
        ui.section("Key takeaways")
        for i in insights:
            with st.expander(f"{i.tag} — {i.headline}", expanded=False):
                st.markdown(i.detail)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEGMENT DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════
with tab_deep:
    available = present_segments(rfm)
    active = st.selectbox("Segment", options=available, key="deep_seg", label_visibility="collapsed")
    row = seg[seg["Segment"] == active].iloc[0]
    seg_data = rfm[rfm["Segment"] == active]
    color = SEGMENT_COLORS[active]
    rec = RECOMMENDATIONS[active]

    ui.render(f"""
    <div class="seg-head">
      <div class="seg-icon">{SEGMENT_ICONS[active]}</div>
      <div><div class="seg-name" style="color:{color};">{ui.esc(active)}</div>
      <div class="page-subtitle" style="margin-top:2px;">{ui.esc(SEGMENT_DESCRIPTIONS[active])}</div></div>
      <div style="margin-left:auto;">{ui.pill(rec['priority'], rec['color'])}</div>
    </div>""")
    ui.spacer("sm")

    mult = row["Revenue_per_Customer"] / VIEW["avg_customer_value"] if VIEW["avg_customer_value"] else 0.0
    deep_kpis = [
        ("Customers", fmt_count(row["Customers"]), f"<b>{row['Customer_Pct']:.1f}%</b> of customers in view"),
        ("Revenue", fmt_money(row["Total_Revenue"], CURRENCY), f"<b>{row['Revenue_Pct']:.1f}%</b> of revenue in view"),
        ("Value per customer", fmt_money(row["Revenue_per_Customer"], CURRENCY),
         f"<b>{mult:.1f}x</b> the average customer"),
        ("Median orders", fmt_count(row["Median_Frequency"]),
         f"median · all customers: <b>{fmt_count(FULL['median_frequency'])}</b>"),
        ("Median days since last order", f"{row['Median_Recency']:.0f}",
         f"median · all customers: <b>{FULL['median_recency']:.0f}</b>"),
    ]
    ui.kpi_row([(label, value, sub, color) for label, value, sub in deep_kpis])

    ui.section(f"How {active} compare with all customers")
    ui.note("Bars show the share of this segment in each value band; the dotted line is the same for all "
            "customers; solid vertical line = segment median, dotted vertical line = overall median. "
            "Orders and spend use log-spaced bands because a few customers are far bigger than the rest.")
    d1, d2, d3 = st.columns(3, gap="large")
    for col, metric, title, sub in [
        (d1, "Recency", "How recently they bought",
         f"Median <b>{row['Median_Recency']:.0f} days</b> since last order (all: {FULL['median_recency']:.0f})"),
        (d2, "Frequency", "How often they order",
         f"Median <b>{fmt_count(row['Median_Frequency'])} orders</b> (all: {fmt_count(FULL['median_frequency'])})"),
        (d3, "Monetary", "How much they spend",
         f"Median <b>{fmt_money(row['Median_Monetary'], CURRENCY)}</b> lifetime (all: "
         f"{fmt_money(FULL['median_monetary'], CURRENCY)})"),
    ]:
        with col:
            ui.chart_header(title, sub, lines=2)
            ui.show_plot(distribution_figure(seg_data[metric], rfm_full[metric], metric, color, CURRENCY),
                         f"dd_{metric}")

    ui.section("Top 10 customers by spend")
    top10 = friendly(seg_data.nlargest(10, "Monetary")[
        ["CustomerID", "Frequency", "Recency", "Monetary", "RFM_Score"]])
    ui.show_table(top10, column_config=TABLE_CONFIG)

    ui.section("Recommended actions")
    tgt = target_customers(rfm, rfm_full, active, PLAYBOOK_PARAMS)
    actions = "".join(
        f'<div class="action"><div class="bullet" style="background:{rec["color"]};"></div>'
        f"<div>{ui.esc(a)}</div></div>" for a in rec["actions"])
    ui.card_grid([
        ui.panel(f"""
          <div class="kpi-accent" style="background:{rec['color']};"></div>
          <div class="panel-title">Priority</div>
          <div style="font-size:1.5rem;font-weight:650;color:{rec['color']};margin-bottom:1rem;">{ui.esc(rec['priority'])}</div>
          <div class="panel-title">Success metric</div>
          <p>{ui.esc(rec['metric'].format(currency=CURRENCY))}</p>
          <div class="panel-title" style="margin-top:1rem;">Priority list</div>
          <p><b>{len(tgt):,}</b> customers · {fmt_money(tgt['Monetary'].sum(), CURRENCY)} historical spend.
          See the Action Playbook tab to size and download it.</p>"""),
        ui.panel(f"""
          <div class="kpi-accent" style="background:{rec['color']};"></div>
          <div class="panel-title">What to do</div>{actions}"""),
    ], "lg", layout="1-2")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACTION PLAYBOOK
# ════════════════════════════════════════════════════════════════════════════
with tab_play:
    pb = build_playbook(rfm, rfm_full, PLAYBOOK_PARAMS)
    ui.section("Priority target groups")
    ui.note("Each segment has one <b>priority group</b>: the customers most worth acting on first. "
            "Thresholds are set in <code>config.yaml</code> under <code>playbook</code>. "
            "Sizes respond to the sidebar filters.")
    show = pd.DataFrame({
        "Segment": pb["Segment"],
        "Priority group": pb["Target group"],
        "Targeted": pb["Targeted customers"],
        "% of segment": (pb["Share of segment"] * 100).map("{:.0f}%".format),
        "Historical spend": pb["Targeted spend"].map(lambda v: fmt_money(v, CURRENCY)),
        "% of segment spend": (pb["Share of segment spend"] * 100).map("{:.0f}%".format),
        "Avg order": pb["Avg order value"].map(lambda v: fmt_money(v, CURRENCY)),
    })
    ui.show_table(show, column_config={
        "Priority group": st.column_config.TextColumn(width="large"),
        "Targeted": st.column_config.NumberColumn(format="%d", width="small"),
        "% of segment": st.column_config.TextColumn(width="small"),
        "% of segment spend": st.column_config.TextColumn("% of spend", width="small"),
        "Historical spend": st.column_config.TextColumn("Spend", width="small"),
        "Avg order": st.column_config.TextColumn(width="small"),
    })

    group_text = dict(zip(pb["Segment"], pb["Target group"]))
    all_targets = pd.concat([
        target_customers(rfm, rfm_full, s, PLAYBOOK_PARAMS).assign(Priority_Group=group_text[s])
        for s in pb["Segment"]
    ])
    all_targets = friendly(all_targets[["CustomerID", "Segment", "Priority_Group", "Recency", "Frequency",
                                        "Monetary", "RFM_Score"]])
    st.download_button("Download all priority lists (CSV)", to_csv(all_targets),
                       file_name="priority_target_lists.csv", mime="text/csv", key="dl_all_targets")

    ui.section("Campaign ROI simulator")
    ui.note("Illustrative economics for contacting one priority group. <b>Margin, discount, cost and response "
            "rate are your assumptions</b> (defaults are placeholders). Each responder is assumed to place one "
            "order at the group's historical average order value; repeat purchases are ignored, so the result "
            "is conservative.")
    c_ctrl, c_out = st.columns([1, 2], gap="large")
    with c_ctrl:
        _sim_options = list(pb["Segment"])
        sim_seg = st.selectbox("Segment to contact", _sim_options, key="sim_seg",
                               index=_sim_options.index("At Risk") if "At Risk" in _sim_options else 0)
        sim = pb[pb["Segment"] == sim_seg].iloc[0]
        resp = st.slider("Expected response rate", 0.0, 50.0, float(ROI_DEFAULTS["response_rate"] * 100),
                         0.5, format="%.1f%%", key="sim_resp") / 100
        cost = st.number_input(f"Cost per contact ({CURRENCY})", 0.0, 100.0,
                               float(ROI_DEFAULTS["cost_per_contact"]), 0.05, key="sim_cost")
        margin = st.slider("Gross margin", 0, 90, round(ROI_DEFAULTS["gross_margin"] * 100),
                           format="%d%%", key="sim_margin") / 100
        disc = st.slider("Discount offered", 0, 60, round(ROI_DEFAULTS["discount"] * 100),
                         format="%d%%", key="sim_disc") / 100
    econ = campaign_economics(int(sim["Targeted customers"]), float(sim["Avg order value"]),
                              resp, cost, margin, disc)
    with c_out:
        net = float(econ["net_profit"])
        roi = econ["roi"]
        be = econ["breakeven_response"]
        sim_cards = [
            ("Contacted", fmt_count(sim["Targeted customers"]), f"avg order {fmt_money(sim['Avg order value'], CURRENCY)}",
             ACCENT),
            ("Expected responders", fmt_count(econ["responders"]), f"at <b>{resp * 100:.1f}%</b> response", ACCENT),
            ("Net profit", fmt_money(net, CURRENCY), "gross profit minus campaign cost", GREEN if net >= 0 else RED),
            ("ROI", ("n/a" if roi is None else ">999%" if roi > 9.99 else f"{roi * 100:.0f}%"),
             (f"break-even response <b>{be * 100:.1f}%</b>" if be <= 1 else "break-even needs <b>&gt;100%</b> response")
             if be is not None else "not profitable at any response",
             GREEN if (roi or 0) >= 0 else RED),
        ]
        ui.kpi_row(sim_cards)
        ui.spacer("sm")
        if be is None:
            verdict = ("Each responder <b>loses money</b>: the discount is at or above the gross margin. "
                       "Lower the discount or this offer cannot pay back.")
        elif be > 1:
            verdict = ("<b>Cannot break even</b>: campaign cost per contact exceeds the profit from a "
                       "responder, so even a 100% response would not cover it.")
        elif net >= 0:
            verdict = (f"<b>Profitable</b> if at least <b>{be * 100:.1f}%</b> respond; you assumed "
                       f"{resp * 100:.1f}%.")
        else:
            verdict = (f"<b>Loses money</b> at {resp * 100:.1f}% response. It needs at least "
                       f"<b>{be * 100:.1f}%</b> to break even.")
        ui.note(verdict)
        ui.show_plot(economics_waterfall(econ, CURRENCY), "sim_waterfall")
    one = friendly(target_customers(rfm, rfm_full, sim_seg, PLAYBOOK_PARAMS)[
        ["CustomerID", "Segment", "Recency", "Frequency", "Monetary", "RFM_Score"]])
    st.download_button(f"Download {sim_seg} priority list (CSV)", to_csv(one),
                       file_name=f"priority_{sim_seg.lower().replace(' / ', '_').replace(' ', '_')}.csv",
                       mime="text/csv", key="dl_one_target")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    ui.section("Customer lookup")
    customer_search = st.text_input("Customer ID", placeholder="Type a Customer ID, e.g. 14688",
                                    key="cid", label_visibility="collapsed")
    found: pd.Series | None = None
    if customer_search.strip():
        try:
            cid = int(customer_search.strip())
        except ValueError:
            st.warning("Enter a valid numeric Customer ID.")
        else:
            match = rfm_full[rfm_full["CustomerID"] == cid]
            if match.empty:
                st.warning(f"Customer ID {cid} not found in dataset.")
            else:
                found = match.iloc[0]

    if found is not None:
        cseg = found["Segment"]
        ccolor = SEGMENT_COLORS.get(cseg, MUTED)
        if not ((rfm["CustomerID"] == found["CustomerID"]).any()):
            st.info("This customer is outside the current sidebar filters, so they are not in the charts below "
                    "but are shown here.")
        aov = found["Monetary"] / max(found["Frequency"], 1)
        profile_cards = [
            ("Segment", f"{SEGMENT_ICONS.get(cseg, '')} {cseg}".strip(),
             f"priority: <b>{ui.esc(priority_label(RECOMMENDATIONS[cseg]['priority']))}</b>"),
            ("Days since last order", f"{int(found['Recency'])}", "since last purchase"),
            ("Orders", f"{int(found['Frequency'])}", f"{fmt_money(aov, CURRENCY)} per order"),
            ("Lifetime spend", fmt_money(found["Monetary"], CURRENCY), "total revenue"),
            ("RFM score", f"{int(found['RFM_Score'])} / 15",
             f"R{int(found['R_Score'])} · F{int(found['F_Score'])} · M{int(found['M_Score'])}"),
        ]
        ui.kpi_row([(label, value, sub, ccolor, label == "Segment") for label, value, sub in profile_cards])
        ui.spacer("md")

        p_left, p_right = st.columns([3, 2], gap="large")
        with p_left:
            ui.chart_header(f"Customer {int(found['CustomerID'])} vs all customers",
                            "Higher bar = better than more customers.", lines=2)
            ui.show_plot(percentile_bars(customer_percentiles(found, rfm_full), ccolor), "cx_percentiles")
        with p_right:
            peers = rfm_full[rfm_full["Segment"] == cseg]
            med_m, med_f, med_r = (peers[c].median() for c in ("Monetary", "Frequency", "Recency"))
            spend_x = f"{found['Monetary'] / med_m:.1f}x" if med_m else "n/a"
            ui.render(f"""
            <div class="panel">
              <div class="panel-title">Versus the typical customer in {ui.esc(cseg)}</div>
              <p>Spend: <b>{spend_x}</b> the median ({fmt_money(med_m, CURRENCY)})</p>
              <p>Orders: <b>{int(found['Frequency'])}</b> vs {fmt_count(med_f)}</p>
              <p>Last order: <b>{int(found['Recency'])}</b> vs {med_r:.0f} days ago</p>
              <div class="panel-title" style="margin-top:0.8rem;">Suggested first step</div>
              <p>{ui.esc(RECOMMENDATIONS[cseg]['actions'][0])}</p>
            </div>""")

    ui.section("Customer map")
    ui.chart_header(
        "Every customer at a glance",
        "Each dot is a customer. <b>Top-left is best</b>: spent a lot and bought recently. "
        "Spend is on a log scale. Hover for details; click legend items to hide segments.")
    ui.show_plot(customer_map(rfm, found, CURRENCY), "cx_map")

    ui.section(f"Customer table · {len(rfm):,} records")
    display_df = friendly(
        rfm.sort_values(["RFM_Score", "Monetary"], ascending=False)[
            ["CustomerID", "Segment", "Recency", "Frequency", "Monetary",
             "R_Score", "F_Score", "M_Score", "RFM_Score"]])
    ui.show_table(display_df, height=420, column_config={
        **TABLE_CONFIG,
        "RFM score": st.column_config.ProgressColumn("RFM score", min_value=SCORE_MIN, max_value=SCORE_MAX,
                                                     format="%d"),
    })
    st.download_button("Download filtered data as CSV", to_csv(display_df),
                       file_name="customer_segments_filtered.csv", mime="text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — METHOD & DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_method:

    ui.section("Why this number of clusters?")
    diag = _load_diagnostics(MTIME)
    k_final = int(CFG["clustering"]["k_final"])
    best_k = int(diag.loc[diag["Silhouette"].idxmax(), "K"])
    in_range = bool((diag["K"] == k_final).any())
    sil_k = float(diag.loc[diag["K"] == k_final, "Silhouette"].iloc[0]) if in_range else float("nan")
    if not in_range:
        sub = (f"The configured K={k_final} is outside the tested range "
               f"({int(diag['K'].min())}-{int(diag['K'].max())}); K={best_k} separates best.")
    elif best_k != k_final:
        sub = (f"Statistically, <b>K={best_k}</b> separates best, but only into {best_k} broad groups. "
               f"<b>K={k_final}</b> (silhouette {sil_k:.2f}) is a <b>business choice</b>: enough tiers to "
               f"run different plays. There is no elbow that proves K={k_final}.")
    else:
        sub = f"K={k_final} has the highest silhouette ({sil_k:.2f}) in the tested range."
    ui.chart_header("Elbow and silhouette by number of clusters", sub)
    ui.show_plot(k_diagnostics(diag, k_final), "mt_k")

    ui.section("RFM score vs segment")
    ui.chart_header(
        "Segments are not score bands",
        "Segments come from clustering raw recency, frequency and spend, not from cutting the RFM score, "
        "so neighbouring segments overlap on score.")
    ui.show_plot(score_distribution(rfm), "mt_scores")

    ui.section("This data run")
    if not META:
        st.info("No run metadata found: `data/rfm_scored.csv` was generated before the latest cleaning "
                "fixes (cancellation netting, accounting-code filter). Run `python rfm_analysis.py` with "
                "`Online Retail.xlsx` in `data/` to refresh it.")
    else:
        stats = META.get("cleaning_stats", {})
        run_items = [
            ("Data as of", str(META.get("snapshot_date", "n/a")), f"first order {META.get('first_transaction', 'n/a')}"),
            ("Customers scored", f"{META.get('customers', 0):,}", "after cleaning"),
            ("Transactions used", f"{META.get('transaction_rows', 0):,}",
             f"{stats.get('duplicate_rows_removed', 0):,} duplicates and "
             f"{stats.get('non_product_rows_removed', 0):,} accounting rows removed"),
            ("Cluster separation", f"{META['silhouette_at_k_final']:.2f}" if META.get("silhouette_at_k_final") is not None else "n/a",
             "silhouette score (1 = perfect)"),
        ]
        for label, value, sub in run_items:
            with st.expander(f"{label}: {value}", expanded=False):
                st.markdown(sub)

    ui.section("Definitions and limits")
    pot = rfm_full[rfm_full["Segment"] == "Potential Loyalists"]
    pot_txt = (f" For example, Potential Loyalists average {pot['Frequency'].mean():.1f} orders here."
               if not pot.empty else "")
    ui.card_grid([
        ui.panel("""<div class="panel-title">Metrics</div>
        <p><b>Recency</b>: days between the customer's last order and the day after the last order in the data.</p>
        <p><b>Frequency</b>: number of distinct invoices.</p>
        <p><b>Monetary</b>: lifetime revenue in the data window, net of cancellations.</p>
        <p><b>RFM score</b>: sum of the three 1-5 quintile scores (3 = weakest, 15 = strongest).</p>"""),
        ui.panel(f"""<div class="panel-title">Read with care</div>
        <p><b>Segments are value tiers.</b> Names reflect rank, not a bespoke behaviour rule.{ui.esc(pot_txt)}</p>
        <p><b>Spend is revenue, not profit</b>, and describes the past, not predicted lifetime value.</p>
        <p><b>One snapshot.</b> It cannot show customers moving between segments over time.</p>"""),
    ], "lg")
