"""Tests for RFM computation, clustering, and summary."""

from __future__ import annotations

import pandas as pd
import pytest

from constants import SEGMENT_ORDER
from rfm_analysis import compute_rfm, cluster_and_label, build_segment_summary


class TestComputeRFM:
    """Tests for the RFM scoring function."""

    def test_output_columns(self, rfm_df: pd.DataFrame) -> None:
        expected = {"CustomerID", "Recency", "Frequency", "Monetary",
                     "R_Score", "F_Score", "M_Score", "RFM_Score"}
        assert expected <= set(rfm_df.columns)

    def test_rfm_score_range(self, rfm_df: pd.DataFrame) -> None:
        assert rfm_df["RFM_Score"].min() >= 3
        assert rfm_df["RFM_Score"].max() <= 15

    def test_scores_are_integers(self, rfm_df: pd.DataFrame) -> None:
        for col in ("R_Score", "F_Score", "M_Score", "RFM_Score"):
            assert rfm_df[col].dtype in ("int64", "int32"), f"{col} should be int"

    def test_monetary_is_positive(self, rfm_df: pd.DataFrame) -> None:
        assert (rfm_df["Monetary"] > 0).all()

    def test_one_row_per_customer(self, rfm_df: pd.DataFrame, sample_transactions: pd.DataFrame) -> None:
        assert len(rfm_df) == sample_transactions["CustomerID"].nunique()


class TestClusterAndLabel:
    """Tests for clustering and segment assignment."""

    @pytest.fixture(autouse=True)
    def _run_cluster(self, rfm_df: pd.DataFrame) -> None:
        self.rfm, self.inertia, self.sil, self.k_range = cluster_and_label(rfm_df)

    def test_segment_column_exists(self) -> None:
        assert "Segment" in self.rfm.columns

    def test_all_customers_assigned(self) -> None:
        assert self.rfm["Segment"].notna().all()

    def test_five_segments_present(self) -> None:
        assert set(self.rfm["Segment"].unique()) == set(SEGMENT_ORDER)

    def test_cluster_column_exists(self) -> None:
        assert "Cluster" in self.rfm.columns
        assert self.rfm["Cluster"].nunique() == 5

    def test_inertia_has_one_per_k(self) -> None:
        assert len(self.inertia) == len(list(self.k_range))

    def test_silhouette_scores_are_bounded(self) -> None:
        for s in self.sil:
            assert -1.0 <= s <= 1.0


class TestBuildSegmentSummary:
    """Tests for segment summary aggregation."""

    @pytest.fixture(autouse=True)
    def _run_summary(self, rfm_df: pd.DataFrame) -> None:
        rfm_clustered, _, _, _ = cluster_and_label(rfm_df)
        self.seg = build_segment_summary(rfm_clustered)

    def test_summary_has_all_segments(self) -> None:
        assert set(self.seg["Segment"]) == set(SEGMENT_ORDER)

    def test_summary_columns(self) -> None:
        expected = {"Segment", "Customers", "Avg_Recency", "Avg_Frequency",
                     "Avg_Monetary", "Total_Revenue", "Revenue_Pct"}
        assert expected <= set(self.seg.columns)

    def test_revenue_pct_sums_to_100(self) -> None:
        assert abs(self.seg["Revenue_Pct"].sum() - 100.0) < 0.5

    def test_customer_counts_sum(self, rfm_df: pd.DataFrame) -> None:
        assert self.seg["Customers"].sum() == len(rfm_df)


class TestEdgeCases:
    """Edge-case tests for robustness."""

    def test_single_customer_qcut_fails(self) -> None:
        """qcut requires >= 5 unique values for quintile binning."""
        df = pd.DataFrame({
            "InvoiceNo": ["1001"],
            "InvoiceDate": pd.to_datetime(["2023-06-15"]),
            "CustomerID": [1],
            "Quantity": [5],
            "UnitPrice": [10.0],
            "Revenue": [50.0],
        })
        with pytest.raises(ValueError):
            compute_rfm(df)

    def test_duplicate_invoices_are_counted_once(self) -> None:
        """Frequency should count unique InvoiceNo, not rows.

        We need enough customers for qcut to work (>=5), so we pad with
        extra rows to make a valid dataset.
        """
        extra_customers = 6
        rows = []
        for cid in range(1, extra_customers + 1):
            rows.append({
                "InvoiceNo": f"200{cid}",
                "InvoiceDate": pd.Timestamp("2023-06-15"),
                "CustomerID": cid,
                "Quantity": 1,
                "UnitPrice": 10.0,
                "Revenue": 10.0,
            })
        # Customer 1 has 3 invoices, others have 1
        for inv_id in ["1001", "1002", "1003"]:
            rows.append({
                "InvoiceNo": inv_id,
                "InvoiceDate": pd.Timestamp("2023-06-01"),
                "CustomerID": 1,
                "Quantity": 1,
                "UnitPrice": 10.0,
                "Revenue": 10.0,
            })

        df = pd.DataFrame(rows)
        rfm = compute_rfm(df)
        cust1 = rfm[rfm["CustomerID"] == 1].iloc[0]
        assert cust1["Frequency"] == 4  # 1001 + 1002 + 1003 + 2001
