"""Streamlit UI helpers: global CSS, HTML components and chart rendering."""

from __future__ import annotations

import html
import inspect
from collections.abc import Sequence
from typing import Any

import streamlit as st

from constants import (
    ACCENT,
    MUTED,
    NAVY,
    NAVY_BORDER,
    NAVY_CARD,
    NAVY_RAISED,
    SUBTLE,
    WHITE,
)

_FONT_STACK = "'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
_MONO_STACK = "'DM Mono', ui-monospace, 'SF Mono', Menlo, monospace"

# (min container width in px, columns) ladders per card count, mobile-first.
_LADDERS: dict[str, dict[int, list[tuple[int, int]]]] = {
    "sm": {6: [(300, 2), (470, 3), (940, 6)], 5: [(300, 2), (470, 3), (790, 5)],
           4: [(300, 2), (720, 4)], 3: [(470, 3)], 2: [(300, 2)]},
    "lg": {4: [(500, 2), (980, 4)], 3: [(740, 3)], 2: [(560, 2)]},
}


def _grid_rules() -> str:
    """Container-query rules mapping card count + size to column count."""
    rules = []
    for size, by_n in _LADDERS.items():
        for n, ladder in by_n.items():
            for width, cols in ladder:
                rules.append(
                    f'  @container (min-width: {width}px) {{ .card-grid[data-n="{n}"][data-size="{size}"] '
                    f"{{ grid-template-columns: repeat({cols}, minmax(0, 1fr)); }} }}")
    # Fixed 1:2 split used by the segment action panels.
    rules.append('  @container (min-width: 640px) { .card-grid[data-n="2"][data-size="lg"][data-layout="1-2"] '
                 "{ grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); } }")
    return "\n".join(rules)


_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300..700&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --bg: __NAVY__;  --card: __CARD__;  --raised: __RAISED__;  --border: __BORDER__;
    --text: __WHITE__;  --muted: __MUTED__;  --subtle: __SUBTLE__;  --accent: __ACCENT__;
    --radius: 12px;  --gap: 1.25rem;
  }

  html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select,
  [data-baseweb="select"], [data-baseweb="tab"] { font-family: __FONT__; }
  .stApp { background: var(--bg); }
  .block-container { padding: 2rem 2.5rem 4rem; max-width: 1440px; }
  @media (max-width: 900px) { .block-container { padding: 1.25rem 1rem 3rem; } }

  /* ── Sidebar ───────────────────────────────────────────── */
  [data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
  [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
  .side-brand { font-size: 1.1rem; font-weight: 650; letter-spacing: -0.01em; color: var(--text); }
  .side-sub { font-size: 0.76rem; color: var(--muted); margin: 2px 0 1.25rem; }
  .side-label { font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
                font-weight: 600; margin: 1.1rem 0 0.35rem; }
  .legend-row { display: flex; align-items: center; justify-content: space-between; gap: 8px;
                font-size: 0.8rem; color: var(--muted); padding: 3px 0; }
  .legend-row .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .legend-row .name { flex: 1; }
  .legend-row .n { font-family: __MONO__; font-size: 0.75rem; color: var(--text); }

  /* ── Header ────────────────────────────────────────────── */
  .page-title { font-size: 1.75rem; font-weight: 650; letter-spacing: -0.025em; color: var(--text);
                line-height: 1.15; }
  .page-subtitle { font-size: 0.9rem; color: var(--muted); margin-top: 0.35rem; }
  .page-head { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between;
               gap: 0.75rem 1.5rem; }
  .page-head > div:first-child { flex: 1 1 320px; min-width: 0; }
  .meta-badge { display: inline-block; max-width: 100%; overflow-wrap: anywhere; font-family: __MONO__; font-size: 0.72rem; color: var(--muted);
                border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px; background: var(--card); }
  .filter-chip { display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--text);
                 background: rgba(110,168,254,0.12); border: 1px solid rgba(110,168,254,0.35);
                 border-radius: 999px; padding: 5px 14px; margin-top: 0.9rem; max-width: 100%; flex-wrap: wrap; }

  /* ── Responsive card grid ──────────────────────────────────
     One grid drives KPI cards, insight cards and panels. Column counts come
     from container queries on the grid's own width (not the viewport), so
     layouts stay correct with the sidebar open or closed. Rows are always
     equal height (grid-auto-rows: 1fr) and wrap into even rows. */
  .cq { container-type: inline-size; width: 100%; }
  .card-grid { display: grid; gap: 0.75rem; grid-template-columns: minmax(0, 1fr); grid-auto-rows: 1fr; }
  .card-grid[data-size="lg"] { gap: var(--gap); }
__GRID_RULES__

  .kpi-card, .insight, .panel { box-sizing: border-box; min-width: 0; height: 100%;
              background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); }
  .kpi-card { container-type: inline-size; display: flex; flex-direction: column;
              padding: 1rem 1.15rem 1.05rem; min-height: var(--kpi-h); }
  :root { --kpi-h: 10.5rem; }
  .kpi-accent { width: 28px; height: 3px; border-radius: 2px; margin-bottom: 0.8rem; flex-shrink: 0; }
  .kpi-label { font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
               font-weight: 600; margin-bottom: 0.4rem; line-height: 1.25; min-height: 2.5em; }
  .kpi-value { font-size: clamp(1.05rem, 11cqw, 1.85rem); font-weight: 650; color: var(--text);
               line-height: 1.1; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
               overflow-wrap: anywhere; }
  .kpi-value.is-text { font-size: clamp(0.95rem, 8cqw, 1.2rem); line-height: 1.2; letter-spacing: -0.01em; }
  .kpi-sub { font-size: 0.78rem; margin-top: auto; padding-top: 0.5rem; color: var(--muted); line-height: 1.35; }
  .kpi-sub b { color: var(--text); font-weight: 600; }

  .insight { border-left-width: 3px; padding: 1rem 1.15rem 1.05rem;
             background: linear-gradient(180deg, var(--card), rgba(17,27,46,0.55)); }
  .insight-tag { font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 650;
                 margin-bottom: 0.45rem; overflow-wrap: break-word; }
  .insight-head { font-size: 1.02rem; font-weight: 600; color: var(--text); line-height: 1.3;
                  overflow-wrap: break-word; }
  .insight-detail { font-size: 0.82rem; color: var(--muted); margin-top: 0.45rem; line-height: 1.45; }

  .panel { padding: 1.1rem 1.3rem; }
  .panel-title { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted);
                 font-weight: 600; margin-bottom: 0.7rem; }
  .panel p, .panel li { font-size: 0.86rem; color: var(--text); line-height: 1.55; }
  .action { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 0.6rem;
            font-size: 0.87rem; color: var(--text); line-height: 1.5; }
  .action .bullet { width: 6px; height: 6px; border-radius: 50%; margin-top: 0.5rem; flex-shrink: 0; }
  .step-num { display: inline-flex; width: 24px; height: 24px; border-radius: 50%; align-items: center;
              justify-content: center; font-size: 0.75rem; font-weight: 650; margin-right: 8px;
              background: rgba(110,168,254,0.15); color: var(--accent); }
  .note { font-size: 0.82rem; color: var(--muted); line-height: 1.5; border-left: 2px solid var(--border);
          padding: 0.15rem 0 0.15rem 0.9rem; margin: 0.6rem 0; }
  .note b { color: var(--text); }

  /* ── Section + chart headers ───────────────────────────── */
  .section-label { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted);
                   font-weight: 650; margin: 2.2rem 0 1rem; padding-bottom: 0.55rem;
                   border-bottom: 1px solid var(--border); }
  .chart-title { font-size: 1.02rem; font-weight: 600; color: var(--text); letter-spacing: -0.005em; }
  .chart-sub { font-size: 0.82rem; color: var(--muted); margin: 0.2rem 0 0.7rem; line-height: 1.45; }
  .chart-sub b { color: var(--text); font-weight: 600; }
  .key-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin: 0 4px 0 2px; }

  .seg-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin: 0.3rem 0 0.2rem; }
  .seg-icon { font-size: 1.7rem; }
  .seg-head > div:nth-child(2) { flex: 1 1 220px; min-width: 0; }
  .seg-name { font-size: 1.35rem; font-weight: 650; letter-spacing: -0.015em; }
  .pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.68rem;
          font-weight: 700; letter-spacing: 0.08em; }

  /* ── Native widget polish ──────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
  .stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; color: var(--muted); font-weight: 550;
                                 font-size: 0.9rem; background: transparent; border-radius: 8px 8px 0 0; }
  .stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.03); }
  .stTabs [aria-selected="true"] { color: var(--text) !important; }
  .stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 3px; }
  .stTabs [data-baseweb="tab-border"] { display: none; }

  [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  [data-testid="stDownloadButton"] button, [data-testid="stButton"] button {
    background: var(--card); color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; font-size: 0.85rem; transition: border-color .15s, background .15s; }
  [data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover {
    background: var(--raised); border-color: var(--accent); color: var(--text); }
  [data-tag], span[data-baseweb="tag"], [data-baseweb="tag"] { background-color: var(--raised) !important;
                          background: var(--raised) !important; color: var(--text) !important;
                          border: 1px solid var(--border); }
  [data-tag] span, [data-baseweb="tag"] span { color: var(--text) !important; }
  /* Spacers: one scale instead of ad-hoc inline heights */
  .sp-sm { height: 0.8rem; } .sp-md { height: 1.2rem; } .sp-lg { height: 1.6rem; }

  /* Native st.columns wrap instead of crushing charts on mid-size screens */
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 1rem; }
  [data-testid="stColumn"] { min-width: min(100%, 300px) !important; }
  .stPlotlyChart, [data-testid="stPlotlyChart"] { max-width: 100%; }
  [data-testid="stDataFrame"] { max-width: 100%; }

  hr { border-color: var(--border); margin: 1.75rem 0; }

  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }

  #MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] { display: none; }
  header[data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebarCollapseButton"] { visibility: visible !important; opacity: 1 !important;
                                            pointer-events: auto !important; }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet."""
    css = (_CSS
           .replace("__NAVY__", NAVY).replace("__CARD__", NAVY_CARD)
           .replace("__RAISED__", NAVY_RAISED).replace("__BORDER__", NAVY_BORDER)
           .replace("__WHITE__", WHITE).replace("__MUTED__", MUTED)
           .replace("__SUBTLE__", SUBTLE).replace("__ACCENT__", ACCENT)
           .replace("__GRID_RULES__", _grid_rules())
           .replace("__FONT__", _FONT_STACK).replace("__MONO__", _MONO_STACK))
    st.markdown(css, unsafe_allow_html=True)


def _flat(markup: str) -> str:
    """Collapse indentation so Markdown never mistakes HTML for a code block.

    ``$`` becomes ``&#36;`` so a ``$`` currency symbol is never read as LaTeX.
    """
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    return flat.replace("$", "&#36;")


def render(markup: str) -> None:
    """Render trusted HTML."""
    st.markdown(_flat(markup), unsafe_allow_html=True)


def esc(text: Any) -> str:
    """HTML-escape any value."""
    return html.escape(str(text))


def spacer(size: str = "md") -> None:
    """Vertical gap from a fixed scale: ``sm`` / ``md`` / ``lg``."""
    render(f'<div class="sp-{size}"></div>')


def page_header(title: str, subtitle: str, badge: str = "") -> None:
    """Responsive page title block with an optional right-hand badge."""
    chip = f'<div><span class="meta-badge">{esc(badge)}</span></div>' if badge else ""
    render(f'''<div class="page-head"><div><div class="page-title">{esc(title)}</div>
    <div class="page-subtitle">{esc(subtitle)}</div></div>{chip}</div>''')


# ── Components ───────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = "", color: str = ACCENT, text: bool = False) -> str:
    """KPI card HTML. ``sub`` may contain trusted inline HTML (<b>).

    ``text=True`` uses the smaller wrapping value style for names (e.g. a segment).
    """
    cls = "kpi-value is-text" if text else "kpi-value"
    return _flat(f"""
    <div class="kpi-card">
      <div class="kpi-accent" style="background:{color};"></div>
      <div class="kpi-label">{esc(label)}</div>
      <div class="{cls}">{esc(value)}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""")


def card_grid(cards: Sequence[str], size: str = "sm", layout: str = "") -> None:
    """Render pre-built card HTML in the shared equal-height responsive grid."""
    attrs = f'data-n="{len(cards)}" data-size="{size}"' + (f' data-layout="{layout}"' if layout else "")
    render(f'<div class="cq"><div class="card-grid" {attrs}>{"".join(cards)}</div></div>')


def kpi_row(cards: Sequence[tuple[Any, ...]]) -> None:
    """Render KPI cards in the responsive grid: same height everywhere, even wrapping.

    Each item is an argument tuple for :func:`kpi_card` (label, value, sub,
    color, text — the last two optional).
    """
    card_grid([kpi_card(*card) for card in cards], "sm")


def insight_card(tag: str, headline: str, detail: str, color: str) -> str:
    """Auto-generated insight card HTML."""
    return _flat(f"""
    <div class="insight" style="border-left-color:{color};">
      <div class="insight-tag" style="color:{color};">{esc(tag)}</div>
      <div class="insight-head">{esc(headline)}</div>
      <div class="insight-detail">{esc(detail)}</div>
    </div>""")


def panel(inner: str) -> str:
    """Plain content panel HTML (use inside :func:`card_grid`)."""
    return _flat(f'<div class="panel">{inner}</div>')


def section(title: str) -> None:
    """Section divider label."""
    render(f'<div class="section-label">{esc(title)}</div>')


def chart_header(title: str, subtitle: str = "", lines: int = 0) -> None:
    """Chart title plus a takeaway. ``subtitle`` may contain trusted inline HTML.

    ``lines`` reserves that many subtitle lines so charts placed side by side
    start at the same height.
    """
    reserve = f' style="min-height:{lines * 1.45 * 0.82:.2f}rem;"' if lines else ""
    sub = f'<div class="chart-sub"{reserve}>{subtitle}</div>' if subtitle else ""
    render(f'<div class="chart-title">{esc(title)}</div>{sub}')


def key_dot(color: str) -> str:
    """Small colour swatch for use inside chart subtitles."""
    return f'<span class="key-dot" style="background:{color};"></span>'


def pill(text: str, color: str) -> str:
    """Rounded label HTML."""
    return (f'<span class="pill" style="background:{color}22;color:{color};'
            f'border:1px solid {color}55;">{esc(text)}</span>')


def note(markup: str) -> None:
    """Muted explanatory note. ``markup`` may contain trusted inline HTML."""
    render(f'<div class="note">{markup}</div>')


# ── Streamlit version compatibility ──────────────────────────────────────
_PLOTLY_PARAMS = inspect.signature(st.plotly_chart).parameters
_DF_PARAMS = inspect.signature(st.dataframe).parameters


def show_plot(fig: Any, key: str) -> None:
    """Render a Plotly figure full-width without the deprecated container flag."""
    kwargs: dict[str, Any] = {"width": "stretch"} if "width" in _PLOTLY_PARAMS else {
        "use_container_width": True}
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **kwargs)


def show_table(df: Any, **kwargs: Any) -> None:
    """Render a dataframe full-width on both old and new Streamlit versions."""
    kwargs.update({"width": "stretch"} if "width" in _DF_PARAMS else {"use_container_width": True})
    st.dataframe(df, hide_index=True, **kwargs)
