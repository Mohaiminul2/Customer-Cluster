"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """Synthetic transaction data covering edge cases."""
    rng = np.random.default_rng(42)
    n = 500
    customer_ids = rng.integers(1001, 1050, size=n)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    invoice_dates = rng.choice(dates, size=n)
    quantities = rng.integers(1, 20, size=n)
    unit_prices = rng.uniform(1.0, 100.0, size=n)

    return pd.DataFrame({
        "InvoiceNo": [f"{100000 + i}" for i in range(n)],
        "InvoiceDate": invoice_dates,
        "CustomerID": customer_ids,
        "Quantity": quantities,
        "UnitPrice": unit_prices,
        "Revenue": quantities * unit_prices,
    })


@pytest.fixture()
def rfm_df(sample_transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM from synthetic data using the pipeline function."""
    from rfm_analysis import compute_rfm
    return compute_rfm(sample_transactions)
