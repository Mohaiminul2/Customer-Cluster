"""Tests for RFM computation, clustering, and summary."""

from __future__ import annotations

import pandas as pd
import pytest

from constants import SEGMENT_ORDER
from rfm_analysis import build_segment_summary, cluster_and_label, compute_rfm


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

    def test_single_customer_graceful_fallback(self) -> None:
        """Fewer than 5 customers: scores scale down instead of crashing."""
        df = pd.DataFrame({
            "InvoiceNo": ["1001"],
            "InvoiceDate": pd.to_datetime(["2023-06-15"]),
            "CustomerID": [1],
            "Quantity": [5],
            "UnitPrice": [10.0],
            "Revenue": [50.0],
        })
        rfm = compute_rfm(df)
        assert len(rfm) == 1
        assert rfm["RFM_Score"].iloc[0] == 3
        assert rfm["R_Score"].iloc[0] == 1
        assert rfm["F_Score"].iloc[0] == 1
        assert rfm["M_Score"].iloc[0] == 1

    def test_three_customers_scale_to_three_bins(self) -> None:
        """With 3 customers each metric is scored 1-3 and ordering is kept."""
        rows = []
        dates = ("2023-06-01", "2023-06-10", "2023-06-20")
        revenues = (100.0, 50.0, 10.0)
        for cid, date, revenue in zip(range(1, 4), dates, revenues):
            rows.append({
                "InvoiceNo": f"100{cid}",
                "InvoiceDate": pd.Timestamp(date),
                "CustomerID": cid,
                "Quantity": 1,
                "UnitPrice": revenue,
                "Revenue": revenue,
            })
        df = pd.DataFrame(rows)
        rfm = compute_rfm(df)
        scored = rfm.set_index("CustomerID")
        assert len(rfm) == 3
        for col in ("R_Score", "F_Score", "M_Score"):
            assert scored[col].min() >= 1
            assert scored[col].max() <= 3
        assert scored.loc[3, "R_Score"] == 3
        assert scored.loc[1, "M_Score"] == 3
        assert rfm["RFM_Score"].min() >= 3
        assert rfm["RFM_Score"].max() <= 9

    def test_empty_dataset_raises_clear_error(self) -> None:
        """No customers left: a clear ValueError instead of a pandas crash."""
        df = pd.DataFrame({
            "InvoiceNo": [],
            "InvoiceDate": pd.to_datetime([]),
            "CustomerID": [],
            "Quantity": [],
            "UnitPrice": [],
            "Revenue": [],
        })
        with pytest.raises(ValueError, match="No customers"):
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


# ---------------------------------------------------------------------------
# Cleaning and labelling fixes
# ---------------------------------------------------------------------------
def _raw(rows: list[tuple]) -> pd.DataFrame:
    base = pd.Timestamp("2011-12-01")
    return pd.DataFrame(
        [{"InvoiceNo": inv, "StockCode": code, "Quantity": q, "UnitPrice": price,
          "CustomerID": cid, "InvoiceDate": base + pd.Timedelta(days=day)}
         for inv, cid, code, q, price, day in rows]
    )


class TestCleanTransactions:
    CFG_ON = {"drop_duplicates": True, "net_cancellations": True,
              "non_product_stock_codes": ["D", "M", "S", "B", "CRUK", "BANK CHARGES", "AMAZONFEE"]}

    def test_accounting_codes_are_filtered_by_stockcode_not_invoice(self) -> None:
        """Regression: the old filter checked InvoiceNo prefixes for D/M/S, which are StockCodes."""
        from rfm_analysis import clean_transactions
        raw = _raw([("500001", 1, "22423", 1, 5.0, 0),
                    ("500002", 1, "M", 1, 500.0, 1),
                    ("500003", 1, "d", 1, 20.0, 2)])
        out = clean_transactions(raw, self.CFG_ON)
        assert list(out["StockCode"]) == ["22423"]

    def test_exact_duplicates_dropped(self) -> None:
        from rfm_analysis import clean_transactions
        raw = _raw([("500001", 1, "22423", 2, 5.0, 0)] * 2)
        assert len(clean_transactions(raw, self.CFG_ON)) == 1
        assert len(clean_transactions(raw, {**self.CFG_ON, "drop_duplicates": False})) == 2

    def test_non_numeric_invoices_other_than_cancellations_dropped(self) -> None:
        from rfm_analysis import clean_transactions
        raw = _raw([("A500001", 1, "22423", 1, 5.0, 0), ("500002", 1, "22423", 1, 5.0, 1)])
        assert list(clean_transactions(raw, self.CFG_ON)["InvoiceNo"]) == ["500002"]

    def test_cancellations_become_negative_revenue_only_when_netting(self) -> None:
        from rfm_analysis import clean_transactions
        raw = _raw([("500001", 1, "22423", 10, 5.0, 0), ("C500002", 1, "22423", -10, 5.0, 3)])
        on = clean_transactions(raw, self.CFG_ON)
        assert sorted(on["Revenue"]) == [-50.0, 50.0]
        off = clean_transactions(raw, {**self.CFG_ON, "net_cancellations": False})
        assert list(off["Revenue"]) == [50.0]

    def test_rows_without_customer_dropped(self) -> None:
        from rfm_analysis import clean_transactions
        raw = _raw([("500001", None, "22423", 1, 5.0, 0), ("500002", 7, "22423", 1, 5.0, 1)])
        assert list(clean_transactions(raw, self.CFG_ON)["CustomerID"]) == [7]


class TestNettingInRFM:
    def _tx(self) -> pd.DataFrame:
        from rfm_analysis import clean_transactions
        raw = _raw([
            ("500001", 1, "A", 1000, 10.0, 0), ("C500002", 1, "A", -1000, 10.0, 5),   # fully cancelled
            ("500003", 2, "A", 10, 10.0, 1), ("C500004", 2, "A", -4, 10.0, 6),         # partly returned
            *[(f"6000{c}", c, "A", 1, 10.0, 2) for c in range(3, 9)],
        ])
        return clean_transactions(raw, TestCleanTransactions.CFG_ON)

    def test_fully_cancelled_customer_is_dropped(self) -> None:
        rfm = compute_rfm(self._tx())
        assert 1 not in set(rfm["CustomerID"])

    def test_partial_return_reduces_spend_not_frequency_or_recency(self) -> None:
        rfm = compute_rfm(self._tx()).set_index("CustomerID")
        assert rfm.loc[2, "Monetary"] == pytest.approx(60.0)
        assert rfm.loc[2, "Frequency"] == 1
        # Recency measured from the last *sale*, so the later return does not refresh it
        snapshot_gap = (pd.Timestamp("2011-12-03") + pd.Timedelta(days=1)) - pd.Timestamp("2011-12-02")
        assert rfm.loc[2, "Recency"] == snapshot_gap.days


class TestLabelling:
    def test_input_not_mutated(self, rfm_df: pd.DataFrame) -> None:
        before = set(rfm_df.columns)
        cluster_and_label(rfm_df)
        assert set(rfm_df.columns) == before

    def test_bad_k_final_gives_clear_error(self, rfm_df: pd.DataFrame, monkeypatch) -> None:
        import rfm_analysis
        cfg = rfm_analysis.load_config()
        cfg["clustering"]["k_final"] = 4
        monkeypatch.setattr(rfm_analysis, "load_config", lambda: cfg)
        with pytest.raises(ValueError, match="k_final"):
            cluster_and_label(rfm_df)

    def test_segments_are_ordered_by_value(self, rfm_df: pd.DataFrame) -> None:
        out, *_ = cluster_and_label(rfm_df)
        by_seg = out.groupby("Segment")[["Frequency", "Monetary"]].mean().reindex(SEGMENT_ORDER)
        value = by_seg["Monetary"].rank() + by_seg["Frequency"].rank()
        assert value.iloc[0] >= value.iloc[-1]

    def test_rank_clusters_is_deterministic_and_complete(self) -> None:
        import numpy as np

        from clustering import rank_clusters
        x = np.array([[0, 2, 2], [0, 2, 2], [1, -1, -1], [1, -1, -1], [0, 0, 0]], dtype=float)
        labels = np.array([7, 7, 3, 3, 5])
        assert rank_clusters(x, labels) == {7: 1, 5: 2, 3: 3}

    def test_stored_segments_reproduce_from_stored_scores(self) -> None:
        """The committed CSV must be exactly what the current clustering logic produces."""
        import os

        from config_loader import BASE_DIR
        stored = pd.read_csv(os.path.join(BASE_DIR, "data", "rfm_scored.csv"))
        base = stored[["CustomerID", "Recency", "Frequency", "Monetary",
                       "R_Score", "F_Score", "M_Score", "RFM_Score"]]
        out, *_ = cluster_and_label(base)
        assert (out["Segment"].to_numpy() == stored["Segment"].to_numpy()).all()
