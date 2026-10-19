"""
Customer Segmentation via RFM Analysis + K-Means Clustering
Dataset: Online Retail (UCI ML Repository)

Usage:
    python rfm_analysis.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from clustering import cap_and_scale, rank_clusters, sweep_k
from config_loader import BASE_DIR, load_config
from constants import (
    AMBER,
    BLUE,
    GREEN,
    GREY,
    NAVY,
    SUBTLE,
    WHITE,
)
from constants import (
    SEGMENT_COLORS as SEG_COLORS,
)
from constants import (
    SEGMENT_ORDER as SEG_ORDER,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("rfm")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT = os.path.join(BASE_DIR, "charts")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Fallback when config.yaml has no ``cleaning`` section.
DEFAULT_NON_PRODUCT_CODES = ("D", "M", "S", "B", "CRUK", "BANK CHARGES", "AMAZONFEE")

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": NAVY, "axes.facecolor": NAVY,
    "axes.edgecolor": SUBTLE, "axes.labelcolor": WHITE,
    "xtick.color": WHITE,     "ytick.color": WHITE,
    "text.color": WHITE,      "grid.color": "#2D3F5F",
    "grid.alpha": 0.4,        "font.family": "DejaVu Sans",
})


# ===========================================================================
# Pipeline functions
# ===========================================================================

def clean_transactions(raw: pd.DataFrame, cleaning: dict[str, Any] | None = None) -> pd.DataFrame:
    """Clean raw Online-Retail-style transactions.

    Steps:
        1. Drop rows without a ``CustomerID`` and (optionally) exact duplicates.
        2. Drop accounting StockCodes (discounts, manual entries, fees, samples...).
           These are *StockCode* values, not InvoiceNo prefixes.
        3. Keep sales (numeric InvoiceNo, Quantity > 0, UnitPrice > 0).
        4. Optionally keep cancellations (InvoiceNo starting with ``C``) as
           negative-revenue rows so :func:`compute_rfm` can net them off spend.

    Returns:
        DataFrame with a ``Revenue`` column (negative for cancellations). Row
        counts removed at each step are stored in ``df.attrs["cleaning_stats"]``.
    """
    if cleaning is None:
        cleaning = load_config().get("cleaning", {})
    net_cancellations = bool(cleaning.get("net_cancellations", True))

    stats: dict[str, int] = {"raw_rows": len(raw)}
    df = raw.dropna(subset=["CustomerID"]).copy()
    stats["rows_without_customer"] = stats["raw_rows"] - len(df)

    if cleaning.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        stats["duplicate_rows_removed"] = before - len(df)

    if "StockCode" in df.columns:
        blocked = {str(c).upper() for c in cleaning.get("non_product_stock_codes",
                                                          DEFAULT_NON_PRODUCT_CODES)}
        is_non_product = df["StockCode"].astype(str).str.upper().isin(blocked)
        stats["non_product_rows_removed"] = int(is_non_product.sum())
        df = df[~is_non_product]

    invoice = df["InvoiceNo"].astype(str)
    is_sale_invoice = invoice.str.fullmatch(r"\d+")
    is_cancel_invoice = invoice.str.upper().str.startswith("C")

    sales = df[is_sale_invoice & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    if net_cancellations:
        cancels = df[is_cancel_invoice & (df["Quantity"] < 0) & (df["UnitPrice"] > 0)]
    else:
        cancels = df.iloc[0:0]
    stats["sales_rows"] = len(sales)
    stats["cancellation_rows"] = len(cancels)

    out = pd.concat([sales, cancels]).copy()
    out["CustomerID"] = out["CustomerID"].astype(int)
    out["Revenue"] = out["Quantity"] * out["UnitPrice"]
    out.attrs["cleaning_stats"] = stats
    return out


def load_and_clean() -> pd.DataFrame:
    """Load the raw Excel dataset and clean it with :func:`clean_transactions`.

    Raises:
        FileNotFoundError: If the Excel file is missing from ``data/``.
    """
    excel_path = os.path.join(DATA_DIR, "Online Retail.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            "Online Retail.xlsx not found. Download it from the UCI ML Repository "
            "and place it under 'data/'."
        )

    raw = pd.read_excel(excel_path, parse_dates=["InvoiceDate"])
    df = clean_transactions(raw)
    stats = df.attrs.get("cleaning_stats", {})
    log.info("Cleaning: %s", stats)
    log.info("Loaded %d records, %d unique customers", len(df), df["CustomerID"].nunique())
    return df


def snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """Reference date for Recency: the day after the last *sale*."""
    sales = df.loc[df["Revenue"] > 0, "InvoiceDate"]
    return sales.max() + pd.Timedelta(days=1)


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-customer RFM scores.

    Frequency and Recency come from sales only. Monetary is net spend: sales
    minus any cancellation rows (negative ``Revenue``). Customers whose net
    spend is zero or negative are dropped. Each metric is binned into
    quintiles (1-5) and summed into ``RFM_Score``; datasets with fewer than
    five customers fall back to fewer score bins instead of raising.

    Args:
        df: Cleaned transactions with ``CustomerID``, ``InvoiceDate``,
            ``InvoiceNo`` and ``Revenue`` columns.

    Returns:
        DataFrame with CustomerID, Recency, Frequency, Monetary, R_Score,
        F_Score, M_Score and RFM_Score columns.

    Raises:
        ValueError: If no customers remain after aggregation.
    """
    sales = df[df["Revenue"] > 0]
    if sales.empty:
        raise ValueError("No customers to score after aggregation.")
    returns = df[df["Revenue"] < 0]

    snapshot = snapshot_date(df)
    grouped = sales.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    grouped["Recency"] = (snapshot - grouped["LastPurchase"]).dt.days
    if not returns.empty:
        refunded = returns.groupby("CustomerID")["Revenue"].sum()
        grouped["Monetary"] = grouped["Monetary"] + refunded.reindex(grouped.index).fillna(0.0)

    rfm = grouped[["Recency", "Frequency", "Monetary"]].reset_index()
    rfm["Monetary"] = rfm["Monetary"].round(2)

    non_positive = int((rfm["Monetary"] <= 0).sum())
    if non_positive:
        log.warning("Dropping %d customers whose cancellations cancel out their spend.",
                    non_positive)
        rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)

    n_bins = min(5, len(rfm))
    if n_bins == 0:
        raise ValueError("No customers to score after aggregation.")
    if n_bins < 5:
        log.warning("Only %d customers: falling back from quintiles to %d score bins.",
                    len(rfm), n_bins)

    if n_bins == 1:
        rfm["R_Score"] = 1
        rfm["F_Score"] = 1
        rfm["M_Score"] = 1
    else:
        r_labels = list(range(n_bins, 0, -1))
        fm_labels = list(range(1, n_bins + 1))
        rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), n_bins, labels=r_labels).astype(int)
        rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), n_bins, labels=fm_labels).astype(int)
        rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), n_bins, labels=fm_labels).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    log.info("RFM scores computed (range %d-%d)", rfm["RFM_Score"].min(), rfm["RFM_Score"].max())
    return rfm


def cluster_and_label(rfm: pd.DataFrame) -> tuple[pd.DataFrame, list[float], list[float], range]:
    """Run K-Means clustering and assign business segment labels.

    Outliers are capped at configured percentiles before scaling. Clusters are
    ranked by their standardised centroid value (see
    :func:`clustering.rank_clusters`) and named best-to-worst after the five
    business segments. The input frame is not modified.

    Args:
        rfm: RFM DataFrame from :func:`compute_rfm`.

    Returns:
        Tuple of (labelled rfm copy, inertia values, silhouette scores, K range).

    Raises:
        ValueError: If ``k_final`` does not match the number of segment labels.
    """
    cfg = load_config()
    clustering = cfg["clustering"]
    if clustering["k_final"] != len(SEG_ORDER):
        raise ValueError(
            f"clustering.k_final={clustering['k_final']} but there are {len(SEG_ORDER)} "
            "segment labels. Set k_final to 5 or extend constants.SEGMENT_ORDER."
        )

    rfm = rfm.copy()
    x_scaled = cap_and_scale(rfm, cfg["caps"])

    k_range = range(clustering["k_range_start"], clustering["k_range_end"] + 1)
    inertia, sil = sweep_k(x_scaled, k_range, clustering["random_state"], clustering["n_init"])

    km_final = KMeans(n_clusters=clustering["k_final"],
                      random_state=clustering["random_state"],
                      n_init=clustering["n_init"])
    rfm["Cluster"] = km_final.fit_predict(x_scaled)

    rank_map = rank_clusters(x_scaled, rfm["Cluster"].to_numpy())
    label_map = {rank: SEG_ORDER[rank - 1] for rank in range(1, len(SEG_ORDER) + 1)}
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
    return rfm, inertia, sil, k_range


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
    seg = seg.sort_values("_ord").drop(columns="_ord")

    log.info("Segment summary:\n%s", seg.to_string(index=False))
    return seg


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure atomically and close it."""
    final_path = os.path.join(OUT, name)
    tmp_path = final_path + ".tmp"
    fig.savefig(tmp_path, format="png", dpi=150, bbox_inches="tight", facecolor=NAVY)
    os.replace(tmp_path, final_path)
    plt.close(fig)
    log.info("Saved %s", name)


def generate_charts(
    rfm: pd.DataFrame,
    seg: pd.DataFrame,
    inertia: list[float],
    sil: list[float],
    k_range: range,
) -> None:
    """Generate all five static PNG charts.

    Args:
        rfm: Labelled RFM DataFrame.
        seg: Segment summary DataFrame.
        inertia: Inertia values per K.
        sil: Silhouette scores per K.
        k_range: Range of K values tested.
    """
    os.makedirs(OUT, exist_ok=True)
    k_final = load_config()["clustering"]["k_final"]
    present = [s for s in SEG_ORDER if s in set(rfm["Segment"])]

    seg_c = [SEG_COLORS[s] for s in seg["Segment"]]
    palette = {s: SEG_COLORS[s] for s in SEG_ORDER}

    # --- 1. Elbow + Silhouette ---
    ks = list(k_range)
    best_k = ks[int(np.argmax(sil))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=NAVY)
    fig.suptitle("How many clusters? (K)", color=WHITE, fontsize=13, fontweight="bold")

    ax1.plot(ks, inertia, "o-", color=BLUE, lw=2)
    ax1.axvline(k_final, color=AMBER, ls="--", alpha=0.8, label=f"chosen K={k_final}")
    ax1.set_title("Elbow: lower = tighter clusters", color=WHITE, fontsize=10)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia (within-cluster spread)")
    ax1.grid(True)
    ax1.legend(facecolor=NAVY, edgecolor=SUBTLE)

    ax2.plot(ks, sil, "o-", color=GREEN, lw=2)
    ax2.axvline(k_final, color=AMBER, ls="--", alpha=0.8, label=f"chosen K={k_final}")
    ax2.set_title("Silhouette: higher = better separated", color=WHITE, fontsize=10)
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette score")
    ax2.grid(True)
    ax2.legend(facecolor=NAVY, edgecolor=SUBTLE)
    if best_k != k_final:
        ax2.annotate(f"Highest silhouette at K={best_k};\nK={k_final} chosen for business granularity",
                     xy=(k_final, sil[ks.index(k_final)]), xytext=(0.35, 0.75),
                     textcoords="axes fraction", fontsize=8, color=WHITE,
                     arrowprops=dict(arrowstyle="->", color=AMBER))

    plt.tight_layout()
    _save(fig, "01_elbow_silhouette.png")

    # --- 2. Segment distribution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=NAVY)
    fig.suptitle("Few customers, most of the revenue", color=WHITE, fontsize=13, fontweight="bold")

    bars = ax1.barh(seg["Segment"], seg["Customers"], color=seg_c, height=0.55, edgecolor="none")
    ax1.set_title("Customers per segment", color=WHITE, fontsize=10)
    ax1.set_xlabel("Customers")
    ax1.invert_yaxis()
    ax1.grid(True, axis="x")
    for b in bars:
        ax1.text(b.get_width() + 2, b.get_y() + b.get_height() / 2,
                 f"{int(b.get_width()):,}", va="center", fontsize=9)

    bars2 = ax2.barh(seg["Segment"], seg["Revenue_Pct"], color=seg_c, height=0.55, edgecolor="none")
    ax2.set_title("Share of total revenue (%)", color=WHITE, fontsize=10)
    ax2.set_xlabel("Revenue %")
    ax2.invert_yaxis()
    ax2.grid(True, axis="x")
    for b in bars2:
        ax2.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                 f"{b.get_width():.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    _save(fig, "02_segment_distribution.png")

    # --- 3. RFM Box plots (log scale where data is heavily skewed) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=NAVY)
    fig.suptitle("Segment profiles: typical customer in each segment", color=WHITE,
                 fontsize=13, fontweight="bold")
    metrics = [
        ("Recency",  "Days since last purchase (lower = better)", False),
        ("Frequency", "Number of orders (log scale)", True),
        ("Monetary",  "Total spend (log scale)", True),
    ]

    for ax, (col, lbl, use_log) in zip(axes, metrics):
        data = [rfm[rfm["Segment"] == s][col].values for s in present]
        bp = ax.boxplot(
            data, patch_artist=True, vert=True,
            medianprops=dict(color=WHITE, lw=2),
            whiskerprops=dict(color=WHITE),
            capprops=dict(color=WHITE),
            flierprops=dict(marker="o", markerfacecolor=GREY, markeredgecolor="none",
                            markersize=2, alpha=0.3),
        )
        for patch, seg_name in zip(bp["boxes"], present):
            patch.set_facecolor(palette[seg_name])
            patch.set_alpha(0.85)
        if use_log:
            ax.set_yscale("log")
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8)
        ax.set_title(lbl, color=WHITE, fontsize=10)
        ax.grid(True, axis="y")

    plt.tight_layout()
    _save(fig, "03_rfm_boxplots.png")

    # --- 4. Scatter ---
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
    ax.set_facecolor(NAVY)
    for s in present:
        sub = rfm[rfm["Segment"] == s]
        ax.scatter(sub["Recency"], sub["Monetary"], c=palette[s], label=s,
                   alpha=0.55, s=18, edgecolors="none")
    ax.set_yscale("log")
    ax.set_xlabel("Days since last purchase  (further right = less recent)")
    ax.set_ylabel("Total spend (log scale)")
    ax.set_title("Where each customer sits: recency vs spend", color=WHITE, fontsize=13,
                 fontweight="bold")
    ax.legend(facecolor=NAVY, edgecolor=SUBTLE, fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    _save(fig, "04_scatter.png")

    # --- 5. Radar chart (average percentile rank; 0.5 = typical customer) ---
    n_dims = 3
    angles = [n / n_dims * 2 * np.pi for n in range(n_dims)]
    angles.append(angles[0])
    pct = pd.DataFrame({
        "Recency": 1 - rfm["Recency"].rank(pct=True),   # inverted: recent = high
        "Frequency": rfm["Frequency"].rank(pct=True),
        "Monetary": rfm["Monetary"].rank(pct=True),
        "Segment": rfm["Segment"],
    })

    fig = plt.figure(figsize=(8, 8), facecolor=NAVY)
    ax  = fig.add_subplot(111, polar=True, facecolor=NAVY)
    ax.spines["polar"].set_color(SUBTLE)
    ax.tick_params(colors=WHITE)

    for s in present:
        vals = pct[pct["Segment"] == s][["Recency", "Frequency", "Monetary"]].mean().tolist()
        vals.append(vals[0])
        ax.plot(angles, vals, color=palette[s], lw=2, label=s)
        ax.fill(angles, vals, color=palette[s], alpha=0.08)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.5 = typical", "0.75"], color=SUBTLE, fontsize=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Recency\n(recent = high)", "Frequency", "Monetary"], color=WHITE, fontsize=10)
    ax.set_title("Segment profiles: average percentile rank", color=WHITE, fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
              facecolor=NAVY, edgecolor=SUBTLE, fontsize=9)
    plt.tight_layout()
    _save(fig, "05_radar.png")

    log.info("All 5 charts generated.")


# ===========================================================================
# Main
# ===========================================================================

def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV atomically (write to a temp file, then rename)."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _atomic_write_json(payload: dict[str, Any], path: str) -> None:
    """Write a JSON document atomically."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def build_metadata(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    sil: list[float],
    k_range: range,
) -> dict[str, Any]:
    """Run facts the dashboard shows as 'data as of' and model diagnostics."""
    cfg = load_config()
    ks = list(k_range)
    k_final = cfg["clustering"]["k_final"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": snapshot_date(df).date().isoformat(),
        "first_transaction": df["InvoiceDate"].min().date().isoformat(),
        "customers": int(len(rfm)),
        "transaction_rows": int(len(df)),
        "k_final": k_final,
        "silhouette_at_k_final": round(sil[ks.index(k_final)], 4) if k_final in ks else None,
        "best_k_by_silhouette": ks[int(np.argmax(sil))],
        "cleaning": cfg.get("cleaning", {}),
        "cleaning_stats": df.attrs.get("cleaning_stats", {}),
    }


def main() -> None:
    """Run the full segmentation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    df = load_and_clean()
    rfm = compute_rfm(df)
    rfm, inertia, sil, k_range = cluster_and_label(rfm)
    seg = build_segment_summary(rfm)
    generate_charts(rfm, seg, inertia, sil, k_range)

    os.makedirs(DATA_DIR, exist_ok=True)
    _atomic_write_csv(rfm, os.path.join(DATA_DIR, "rfm_scored.csv"))
    _atomic_write_csv(seg, os.path.join(DATA_DIR, "segment_summary.csv"))
    _atomic_write_json(build_metadata(df, rfm, sil, k_range),
                       os.path.join(DATA_DIR, "run_metadata.json"))
    log.info("Data saved. Segment distribution:\n%s", rfm["Segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
