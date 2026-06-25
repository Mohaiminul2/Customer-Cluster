"""
Customer Segmentation via RFM Analysis + K-Means Clustering
Dataset: Online Retail (UCI ML Repository)

Usage:
    python rfm_analysis.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from constants import (
    NAVY, WHITE, BLUE, TEAL, AMBER, RED, GREEN, GREY,
    SEGMENT_COLORS as SEG_COLORS,
    SEGMENT_ORDER as SEG_ORDER,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rfm")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "charts"), exist_ok=True)
OUT = os.path.join(BASE_DIR, "charts")

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
def _load_config(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml from the project root."""
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

CFG = _load_config()

K_FINAL: int = CFG["clustering"]["k_final"]
RANDOM_STATE: int = CFG["clustering"]["random_state"]
N_INIT: int = CFG["clustering"]["n_init"]
K_START: int = CFG["clustering"]["k_range_start"]
K_END: int = CFG["clustering"]["k_range_end"]

MON_CAP_Q: float = CFG["caps"]["monetary_quantile"]
FREQ_CAP_Q: float = CFG["caps"]["frequency_quantile"]
REC_CAP_Q: float = CFG["caps"]["recency_quantile"]

REC_BINS: list[int] = CFG["heatmap"]["recency_bins"]
REC_LABELS: list[str] = CFG["heatmap"]["recency_labels"]
MON_BINS: list[int] = CFG["heatmap"]["monetary_bins"]
MON_LABELS: list[str] = CFG["heatmap"]["monetary_labels"]

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": NAVY,
    "axes.edgecolor": WHITE,  "axes.labelcolor": WHITE,
    "xtick.color": WHITE,     "ytick.color": WHITE,
    "text.color": WHITE,      "grid.color": "#2D3F5F",
    "grid.alpha": 0.4,        "font.family": "DejaVu Sans",
})


# ===========================================================================
# Pipeline functions
# ===========================================================================

def load_and_clean() -> pd.DataFrame:
    """Load the raw Excel dataset and apply cleaning filters.

    Returns:
        Cleaned DataFrame with a ``Revenue`` column appended.

    Raises:
        FileNotFoundError: If the Excel file is missing from ``data/``.
    """
    excel_path = os.path.join(BASE_DIR, "data", "Online Retail.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            "Online Retail.xlsx not found. Download it from the UCI ML Repository "
            "and place it under 'data/'."
        )

    df = pd.read_excel(excel_path, parse_dates=["InvoiceDate"])
    df.dropna(subset=["CustomerID"], inplace=True)

    # Filter out cancellations, returns, adjustments (InvoiceNo prefixes)
    df["InvoiceNo_str"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo_str"].str.startswith(("C", "D", "M", "S"))]
    df.drop(columns=["InvoiceNo_str"], inplace=True)

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    log.info("Loaded %d records, %d unique customers", len(df), df["CustomerID"].nunique())
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-customer RFM scores.

    Each metric is binned into quintiles (1-5) and summed into ``RFM_Score``.

    Args:
        df: Cleaned transaction data with ``CustomerID``, ``InvoiceDate``,
            ``InvoiceNo``, and ``Revenue`` columns.

    Returns:
        DataFrame indexed by ``CustomerID`` with Recency, Frequency, Monetary,
        R_Score, F_Score, M_Score, and RFM_Score columns.
    """
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum"),
    ).reset_index()
    rfm["Monetary"] = rfm["Monetary"].round(2)

    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    log.info("RFM scores computed (range %d-%d)", rfm["RFM_Score"].min(), rfm["RFM_Score"].max())
    return rfm


def cluster_and_label(rfm: pd.DataFrame) -> tuple[pd.DataFrame, list[float], list[float], range]:
    """Run K-Means clustering and assign business segment labels.

    Outliers are capped at configured percentiles before scaling. Clusters are
    ranked by median RFM_Score (descending) then median Monetary (descending)
    and mapped to five business segments.

    Args:
        rfm: RFM DataFrame from :func:`compute_rfm`.

    Returns:
        Tuple of (labelled rfm, inertia values, silhouette scores, K range).
    """
    MON_CAP = rfm["Monetary"].quantile(MON_CAP_Q)
    FREQ_CAP = rfm["Frequency"].quantile(FREQ_CAP_Q)
    REC_CAP = rfm["Recency"].quantile(REC_CAP_Q)

    rfm_capped = rfm[["Recency", "Frequency", "Monetary"]].copy()
    rfm_capped["Monetary"]  = rfm_capped["Monetary"].clip(upper=MON_CAP)
    rfm_capped["Frequency"] = rfm_capped["Frequency"].clip(upper=FREQ_CAP)
    rfm_capped["Recency"]   = rfm_capped["Recency"].clip(upper=REC_CAP)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_capped)

    K_RANGE = range(K_START, K_END + 1)
    inertia: list[float] = []
    sil: list[float] = []
    for k in K_RANGE:
        km  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, lbl))

    km_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=N_INIT)
    rfm["Cluster"] = km_final.fit_predict(X_scaled)

    rank_df = (
        rfm.groupby("Cluster")
        .agg(med_rfm=("RFM_Score", "median"), med_mon=("Monetary", "median"))
        .reset_index()
        .sort_values(["med_rfm", "med_mon"], ascending=[False, False])
    )
    rank_df["rank"] = range(1, len(rank_df) + 1)
    rank_map = rank_df.set_index("Cluster")["rank"].to_dict()

    label_map = {
        1: "Champions",
        2: "Loyal Customers",
        3: "Potential Loyalists",
        4: "At Risk",
        5: "Lost / Churned",
    }
    rfm["Segment"] = rfm["Cluster"].map(rank_map).map(label_map)

    unmapped = rfm["Segment"].isna().sum()
    if unmapped > 0:
        raise ValueError(
            f"{unmapped} customers could not be assigned a segment. "
            "Check cluster count and label mapping."
        )

    seg_counts = rfm["Segment"].value_counts()
    empty_segs = [s for s in SEG_ORDER if s not in seg_counts.index]
    if empty_segs:
        log.warning("Segments with zero customers: %s", empty_segs)

    log.info("Cluster -> Segment mapping:\n%s",
             rfm.groupby(["Cluster", "Segment"])["RFM_Score"].mean().round(1).to_string())
    return rfm, inertia, sil, K_RANGE


def build_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-segment KPIs.

    Args:
        rfm: Labelled RFM DataFrame from :func:`cluster_and_label`.

    Returns:
        Segment summary DataFrame sorted by :data:`SEG_ORDER`.
    """
    seg = (
        rfm.groupby("Segment")
        .agg(
            Customers     = ("CustomerID", "count"),
            Avg_Recency   = ("Recency",    "mean"),
            Avg_Frequency = ("Frequency",  "mean"),
            Avg_Monetary  = ("Monetary",   "mean"),
            Total_Revenue = ("Monetary",   "sum"),
        )
        .round(1)
        .reset_index()
    )
    seg["Revenue_Pct"] = (seg["Total_Revenue"] / seg["Total_Revenue"].sum() * 100).round(1)

    seg["_ord"] = seg["Segment"].map({s: i for i, s in enumerate(SEG_ORDER)})
    seg.sort_values("_ord", inplace=True)
    seg.drop(columns="_ord", inplace=True)

    log.info("Segment summary:\n%s", seg.to_string(index=False))
    return seg


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    log.info("Saved %s", name)


def generate_charts(
    rfm: pd.DataFrame,
    seg: pd.DataFrame,
    inertia: list[float],
    sil: list[float],
    K_RANGE: range,
) -> None:
    """Generate all five static PNG charts.

    Args:
        rfm: Labelled RFM DataFrame.
        seg: Segment summary DataFrame.
        inertia: Inertia values per K.
        sil: Silhouette scores per K.
        K_RANGE: Range of K values tested.
    """
    seg_c = [SEG_COLORS[s] for s in seg["Segment"]]
    palette = {s: SEG_COLORS[s] for s in SEG_ORDER}

    # --- 1. Elbow + Silhouette ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=NAVY)
    fig.suptitle("Optimal Cluster Count (K)", color=WHITE, fontsize=13, fontweight="bold")

    ax1.plot(list(K_RANGE), inertia, "o-", color=BLUE, lw=2)
    ax1.axvline(K_FINAL, color=AMBER, ls="--", alpha=0.7, label=f"K={K_FINAL}")
    ax1.set_title("Elbow - Inertia", color=WHITE)
    ax1.set_xlabel("K")
    ax1.grid(True)
    ax1.legend(facecolor=NAVY, edgecolor=WHITE)

    ax2.plot(list(K_RANGE), sil, "o-", color=GREEN, lw=2)
    ax2.axvline(K_FINAL, color=AMBER, ls="--", alpha=0.7, label=f"K={K_FINAL}")
    ax2.set_title("Silhouette Score", color=WHITE)
    ax2.set_xlabel("K")
    ax2.grid(True)
    ax2.legend(facecolor=NAVY, edgecolor=WHITE)

    plt.tight_layout()
    _save(fig, "01_elbow_silhouette.png")

    # --- 2. Segment distribution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=NAVY)
    fig.suptitle("Segment Distribution", color=WHITE, fontsize=13, fontweight="bold")

    bars = ax1.barh(seg["Segment"], seg["Customers"], color=seg_c, height=0.55, edgecolor="none")
    ax1.set_title("Customer Count", color=WHITE)
    ax1.set_xlabel("Customers")
    ax1.invert_yaxis()
    ax1.grid(True, axis="x")
    for b in bars:
        ax1.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                 f"{int(b.get_width()):,}", va="center", fontsize=9)

    bars2 = ax2.barh(seg["Segment"], seg["Revenue_Pct"], color=seg_c, height=0.55, edgecolor="none")
    ax2.set_title("Revenue Share (%)", color=WHITE)
    ax2.set_xlabel("Revenue %")
    ax2.invert_yaxis()
    ax2.grid(True, axis="x")
    for b in bars2:
        ax2.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                 f"{b.get_width():.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "02_segment_distribution.png")

    # --- 3. RFM Box plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=NAVY)
    fig.suptitle("RFM Distributions by Segment", color=WHITE, fontsize=13, fontweight="bold")
    metrics = [
        ("Recency",  "Days Since Last Purchase"),
        ("Frequency", "# Transactions"),
        ("Monetary",  "Total Spend"),
    ]

    for ax, (col, lbl) in zip(axes, metrics):
        data = [rfm[rfm["Segment"] == s][col].values for s in SEG_ORDER]
        bp = ax.boxplot(
            data, patch_artist=True, vert=True,
            medianprops=dict(color=WHITE, lw=2),
            whiskerprops=dict(color=WHITE),
            capprops=dict(color=WHITE),
            flierprops=dict(marker="o", color=GREY, markersize=2, alpha=0.3),
        )
        for patch, seg_name in zip(bp["boxes"], SEG_ORDER):
            patch.set_facecolor(palette[seg_name])
            patch.set_alpha(0.85)
        ax.set_xticklabels(SEG_ORDER, rotation=30, ha="right", fontsize=7)
        ax.set_title(lbl, color=WHITE, fontsize=10)
        ax.grid(True, axis="y")

    plt.tight_layout()
    _save(fig, "03_rfm_boxplots.png")

    # --- 4. Scatter ---
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
    ax.set_facecolor(NAVY)
    for s in SEG_ORDER:
        sub = rfm[rfm["Segment"] == s]
        ax.scatter(sub["Recency"], sub["Monetary"], c=palette[s], label=s,
                   alpha=0.55, s=18, edgecolors="none")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Total Spend")
    ax.set_title("Recency vs Monetary by Segment", color=WHITE, fontsize=13, fontweight="bold")
    ax.legend(facecolor=NAVY, edgecolor=WHITE, fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    _save(fig, "04_scatter.png")

    # --- 5. Radar chart ---
    N = 3
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    rfm_n = rfm.copy()
    for c in ["Recency", "Frequency", "Monetary"]:
        mn, mx = rfm_n[c].min(), rfm_n[c].max()
        rfm_n[c] = (rfm_n[c] - mn) / (mx - mn)
    rfm_n["Recency"] = 1 - rfm_n["Recency"]

    fig = plt.figure(figsize=(8, 8), facecolor=NAVY)
    ax  = fig.add_subplot(111, polar=True, facecolor=NAVY)
    ax.spines["polar"].set_color(WHITE)
    ax.tick_params(colors=WHITE)

    for s in SEG_ORDER:
        vals = rfm_n[rfm_n["Segment"] == s][["Recency", "Frequency", "Monetary"]].mean().tolist() + [0]
        vals[-1] = vals[0]
        ax.plot(angles, vals, color=palette[s], lw=2, label=s)
        ax.fill(angles, vals, color=palette[s], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Recency\n(inverted)", "Frequency", "Monetary"], color=WHITE, fontsize=10)
    ax.set_title("Segment Profiles - Radar", color=WHITE, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
              facecolor=NAVY, edgecolor=WHITE, fontsize=9)
    plt.tight_layout()
    _save(fig, "05_radar.png")

    log.info("All 5 charts generated.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run the full segmentation pipeline."""
    df = load_and_clean()
    rfm = compute_rfm(df)
    rfm, inertia, sil, K_RANGE = cluster_and_label(rfm)
    seg = build_segment_summary(rfm)
    generate_charts(rfm, seg, inertia, sil, K_RANGE)

    rfm.to_csv(os.path.join(BASE_DIR, "data", "rfm_scored.csv"), index=False)
    seg.to_csv(os.path.join(BASE_DIR, "data", "segment_summary.csv"), index=False)
    log.info("Data saved. Segment distribution:\n%s", rfm["Segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
