# Customer Segmentation Dashboard

An end-to-end customer segmentation system that groups shoppers into actionable business segments using RFM analysis and K-Means clustering, presented through an interactive dark-themed dashboard.

---

## Table of Contents

- [What This Does](#what-this-does)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Dashboard Walkthrough](#dashboard-walkthrough)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Testing](#testing)
- [Future Work](#future-work)
- [License](#license)

---

## What This Does

**The problem:** A retail business has thousands of customers but treats them all the same — sending the same emails, the same offers, the same experience.

**The solution:** This project automatically groups customers into five distinct segments based on their purchasing behaviour, then surfaces those segments in an interactive dashboard so marketing and operations teams can tailor their approach for each group.

### The Five Segments

| Segment | What It Means | Business Action |
|---|---|---|
| **Champions** | Your best customers — recent, frequent, high spenders | Protect them with VIP treatment |
| **Loyal Customers** | Consistent buyers with strong lifetime value | Grow them toward Champions |
| **Potential Loyalists** | Newer customers showing promising patterns | Nurture with onboarding journeys |
| **At Risk** | Previously active customers going quiet | Win back before they churn |
| **Lost / Churned** | Customers who haven't purchased in a long time | Assess whether re-engagement is worth the cost |

### Key Numbers

| Metric | Value |
|---|---|
| Total Customers | 4,338 |
| Total Revenue | ~£8.9M |
| Avg Customers per Segment | ~868 |
| Dataset Source | UCI Online Retail |

---

## Quick Start

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Download the dataset

```bash
curl -L -o data/Online\ Retail.xlsx \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
```

Or download manually from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail) and place the file at `data/Online Retail.xlsx`.

### 3. Run the pipeline (generates data files)

```bash
python3 rfm_analysis.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens at [http://localhost:8501](http://localhost:8501).

---

## How It Works

The system has two components that work independently:

### The Pipeline (`rfm_analysis.py`)

This is the offline data science engine. It runs once to process raw transaction data into structured customer segments.

```
Raw Excel data
    ↓
Data cleaning (remove cancellations, negatives, missing IDs)
    ↓
RFM scoring (bin each customer into quintiles for R, F, M)
    ↓
K-Means clustering (group similar customers together)
    ↓
Segment labelling (map clusters to business names)
    ↓
CSV files + charts (consumed by the dashboard)
```

### The Dashboard (`app.py`)

This is the interactive front-end. It reads the pre-processed CSV files and renders everything as live, filterable charts — no recomputation needed.

---

## Dashboard Walkthrough

### Sidebar Controls

| Control | What It Does |
|---|---|
| Segment multiselect | Show only specific segments across all charts |
| RFM Score slider | Filter customers by their combined RFM score |
| Monetary slider | Filter by total lifetime spend |
| Customer ID search | Look up an individual customer's profile |

### Tab 1 — Overview

High-level portfolio view:

- **Customer count by segment** — horizontal bar chart showing how many customers fall into each group, with percentage labels
- **Revenue share by segment** — donut chart showing which segments drive the most money
- **Recency vs Monetary heatmap** — density view showing where customers cluster on the two most important axes

### Tab 2 — Segment Deep Dive

Select any segment to see:

- **KPI cards** — customer count, average recency, frequency, spend, and revenue share
- **RFM distribution box plots** — how this segment compares to the full portfolio across all three metrics
- **Recommended actions** — pre-built strategy cards with priority level, specific actions, and success metrics for each segment

### Tab 3 — Customer Explorer

- **Individual profile lookup** — enter a customer ID to see their segment, RFM scores, and lifetime value
- **Sortable data table** — view and sort all customers in the filtered set
- **CSV download** — export the current filtered view
- **RFM score histogram** — stacked distribution showing how scores spread across segments

### Design Language

- Dark navy background (`#0F1E35`) with card-based layout
- DM Sans / DM Mono typography
- Segment colour coding: Green → Teal → Blue → Amber → Red (best to worst)
- All charts are interactive Plotly visualisations (hover, zoom, pan)

---

## Configuration

Tune the pipeline without editing code — all key parameters live in `config.yaml`:

```yaml
clustering:
  k_final: 5               # Number of customer segments
  random_state: 42          # Reproducibility seed
  n_init: 20                # K-Means initialisations
  k_range_start: 2          # Elbow analysis range start
  k_range_end: 8            # Elbow analysis range end

caps:
  monetary_quantile: 0.99   # Cap top 1% of spenders for stability
  frequency_quantile: 0.99  # Cap top 1% of frequent buyers
  recency_quantile: 0.99    # Cap top 1% of recent buyers

heatmap:
  recency_bins: [0, 30, 90, 150, 210, 270]      # Days since last purchase
  monetary_bins: [0, 500, 2000, 5000, 20000]     # Lifetime spend bands
```

Changing `k_final` to 4 or 6 will re-cluster customers into that many segments. The label mapping in `rfm_analysis.py` would need updating to match.

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard (interactive UI)
├── rfm_analysis.py         # Offline ML pipeline — RFM scoring, K-Means
├── config.yaml             # Tunable pipeline parameters
├── constants.py            # Shared colours, segment order, icons
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata & tool config
├── data/
│   ├── Online Retail.xlsx  # Raw transaction dataset (23 MB, gitignored)
│   ├── rfm_scored.csv      # Per-customer RFM scores + cluster assignments
│   └── segment_summary.csv # Aggregated metrics per segment
└── tests/
    ├── conftest.py         # Shared pytest fixtures
    ├── test_config.py      # Config loading tests
    ├── test_constants.py   # Constants integrity tests
    └── test_rfm.py         # RFM scoring, clustering, edge cases
```

### What Each File Does

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard — loads CSVs, renders charts, handles user interaction |
| `rfm_analysis.py` | The data pipeline — reads raw Excel, cleans data, runs RFM scoring and K-Means, writes CSVs |
| `config.yaml` | All tuneable parameters in one place — cluster count, outlier caps, heatmap bins |
| `constants.py` | Single source of truth for colours, segment names, and icons used by both `app.py` and `rfm_analysis.py` |
| `requirements.txt` | Pinned Python dependencies for `pip install` |
| `pyproject.toml` | Project metadata, ruff linter config, pytest config |

---

## Technical Details

### Data Cleaning

The raw UCI Online Retail dataset contains ~541,000 transaction rows. Before analysis:

- Rows with missing `CustomerID` are dropped
- Invoice numbers starting with `C` (cancellations), `D`, `M`, or `S` (adjustments) are removed
- Rows with `Quantity <= 0` or `UnitPrice <= 0` are filtered out
- `Revenue = Quantity x UnitPrice` is computed per row

**Result:** ~397,000 clean transactions from 4,338 unique customers.

### RFM Scoring

Each customer is scored on three dimensions using quintile binning:

| Metric | How It's Measured | Score Range |
|---|---|---|
| **Recency** | Days since last purchase (lower is better) | 1-5 (5 = most recent) |
| **Frequency** | Number of unique invoices | 1-5 (5 = most frequent) |
| **Monetary** | Total lifetime spend | 1-5 (5 = highest spender) |

`RFM_Score = R_Score + F_Score + M_Score` (range: 3-15)

### K-Means Clustering

- Outliers capped at the 99th percentile before clustering (configurable in `config.yaml`)
- Features standardised with `StandardScaler` (zero mean, unit variance)
- K=5 chosen via elbow method and silhouette score analysis
- Clusters ranked by median RFM_Score, then median Monetary, to assign business labels

### Segment Summary

After clustering, each segment is profiled:

| Column | Description |
|---|---|
| `Customers` | Number of customers in the segment |
| `Avg_Recency` | Mean days since last purchase |
| `Avg_Frequency` | Mean number of orders |
| `Avg_Monetary` | Mean lifetime spend |
| `Total_Revenue` | Sum of all customer spend in the segment |
| `Revenue_Pct` | Segment's share of total revenue |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_rfm.py -v
```

### Test Coverage

| Test File | What It Covers |
|---|---|
| `test_config.py` | Config loading, structure validation, percentile ranges |
| `test_constants.py` | Segment order, colour/icon coverage, consistency |
| `test_rfm.py` | RFM scoring (columns, ranges, types), clustering (segment assignment, count), summary (aggregation, revenue %), edge cases (single customer, duplicate invoices) |

---

## Future Work

### Short Term

- **[ ] Add streaming data support** — Accept new transactions incrementally instead of reprocessing the entire dataset from scratch
- **[ ] Improve edge-case handling** — Graceful fallback when dataset has fewer than 5 customers (currently `qcut` requires at least 5 data points)
- **[ ] Add segment change tracking** — Compare segment assignments across two time periods to identify customers moving between segments

### Medium Term

- **[ ] External database support** — Connect directly to a SQL database or data warehouse instead of requiring a static Excel file
- **[ ] A/B test integration** — Link segment assignments to marketing campaign performance data to measure ROI per segment
- **[ ] Predictive churn scoring** — Add a binary classifier that predicts which "At Risk" customers are most likely to churn in the next 30/60/90 days
- **[ ] Multi-product segmentation** — Extend RFM to include product category preferences, enabling segment-specific product recommendations

### Long Term

- **[ ] Real-time dashboard** — Replace static CSV loading with live database queries and auto-refresh
- **[ ] Automated retraining pipeline** — Schedule regular re-clustering as new data arrives, with drift detection to alert when segments shift significantly
- **[ ] Customer lifetime value (CLV) prediction** — Build a probabilistic model (BG/NBD or Pareto/NBD) to forecast future customer value on top of historical RFM
- **[ ] Multi-brand / multi-store support** — Extend the schema to handle multiple business units with independent or shared segmentation

---

## Dataset

**UCI Online Retail Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Records: ~541,000 transactions
- Period: December 2010 - December 2011
- Origin: UK-based online retail store
- Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit>=1.32.0` | Dashboard UI framework |
| `pandas>=2.0.0` | Data manipulation |
| `numpy>=1.24.0` | Numerical operations |
| `plotly>=5.18.0` | Interactive charts |
| `scikit-learn>=1.3.0` | K-Means clustering, scaling, silhouette scoring |
| `openpyxl>=3.1.0` | Excel file reading |
| `matplotlib>=3.7.0` | Static chart generation (pipeline only) |
| `pyyaml>=6.0` | Configuration file parsing |


# 1. Download the dataset (manually or via curl — see README)
# 2. Install dependencies
pip install -r requirements.txt
# 3. Generate the CSV files
python3 rfm_analysis.py
# 4. Launch the dashboard
streamlit run app.py