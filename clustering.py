"""Shared clustering helpers used by both the pipeline and the dashboard.

Pure functions only: no plotting and no file I/O, so they are cheap to import
and easy to test.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES: list[str] = ["Recency", "Frequency", "Monetary"]
_CAP_KEYS: dict[str, str] = {
    "Recency": "recency_quantile",
    "Frequency": "frequency_quantile",
    "Monetary": "monetary_quantile",
}
_SILHOUETTE_SAMPLE_LIMIT = 10_000


def cap_and_scale(rfm: pd.DataFrame, caps: dict[str, float]) -> np.ndarray:
    """Cap outliers at the configured quantiles, then standardise R, F and M."""
    capped = rfm[FEATURES].copy()
    for col in FEATURES:
        upper = capped[col].quantile(caps[_CAP_KEYS[col]])
        capped[col] = capped[col].clip(upper=upper)
    return StandardScaler().fit_transform(capped)


def sweep_k(
    x_scaled: np.ndarray,
    k_range: Iterable[int],
    random_state: int,
    n_init: int,
) -> tuple[list[float], list[float]]:
    """Fit K-Means for each K and return (inertia, silhouette) lists."""
    inertia: list[float] = []
    sil: list[float] = []
    sample_size = _SILHOUETTE_SAMPLE_LIMIT if len(x_scaled) > _SILHOUETTE_SAMPLE_LIMIT else None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(x_scaled)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(
            x_scaled, labels, sample_size=sample_size, random_state=random_state,
        )))
    return inertia, sil


def rank_clusters(x_scaled: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Rank clusters by value, best first (rank 1 = most valuable).

    Value is the cluster's mean standardised ``Frequency + Monetary - Recency``.
    Unlike a median of integer RFM scores this is continuous, so ties between
    clusters do not occur in practice and the ordering is stable.
    """
    frame = pd.DataFrame(x_scaled, columns=FEATURES)
    frame["Cluster"] = np.asarray(labels)
    centroids = frame.groupby("Cluster")[FEATURES].mean()
    value = centroids["Frequency"] + centroids["Monetary"] - centroids["Recency"]
    order = value.sort_values(ascending=False, kind="stable").index
    return {int(cluster): rank for rank, cluster in enumerate(order, start=1)}


def diagnostics_table(rfm: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Elbow / silhouette table for the configured K range (used by the dashboard)."""
    clustering = cfg["clustering"]
    x_scaled = cap_and_scale(rfm, cfg["caps"])
    ks = list(range(clustering["k_range_start"], clustering["k_range_end"] + 1))
    inertia, sil = sweep_k(x_scaled, ks, clustering["random_state"], clustering["n_init"])
    return pd.DataFrame({"K": ks, "Inertia": inertia, "Silhouette": sil})
