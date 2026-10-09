# Customer Cluster

An end-to-end customer segmentation system using **RFM analysis** and **K-Means clustering** and presented through a Python web dashboard.

---

**The problem:** 
A retail business has thousands of customers but treats them all the same — sending the same emails, the same offers, the same experience.

**The solution:** 
This project automatically groups customers into five distinct segments based on their purchasing behaviour, then surfaces those segments in an interactive dashboard so marketing and operations teams can tailor their approach for each group.


## Getting Started

### Prerequisites
- Python 3.10 or newer
- The [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) from the UCI ML Repository — download `Online Retail.xlsx` and place it in `data/`

### Installation
```bash
git clone https://github.com/Mohaiminul2/Customer-Cluster.git
cd Customer-Cluster
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # reproducible install (generated lockfile)
# or, for development (adds pytest, pytest-cov and ruff):
pip install -e .[dev]
```

### Generate the segment data
```bash
python rfm_analysis.py
```
This cleans the raw transactions, computes RFM scores, runs K-Means clustering and writes `data/rfm_scored.csv`, `data/segment_summary.csv` and five charts under `charts/`.

### Launch the dashboard
```bash
streamlit run app.py
```

### Tests & lint
```bash
pytest
ruff check .
```


**The Five Segments**

| Segment | What It Means | Business Action |
|---|---|---|
| Champions | Your best customers — recent, frequent, high spenders | Protect them with VIP treatment |
| Loyal Customers | Consistent buyers with strong lifetime value | Grow them toward Champions |
| Potential Loyalists | Newer customers showing promising patterns | Nurture with onboarding journeys |
| At Risk | Previously active customers going quiet | Win back before they churn |
| Lost / Churned | Customers who haven't purchased in a long time | Assess whether re-engagement is worth the cost |



**RFM Scoring**

Each customer is scored on three dimensions using quintile binning:

| Metric | How It's Measured | Score Range |
|---|---|---|
| Recency | Days since last purchase (lower is better) | 1-5 (5 = most recent) |
| Frequency | Number of unique invoices | 1-5 (5 = most frequent) |
| Monetary | Total lifetime spend | 1-5 (5 = highest spender) |

`RFM_Score = R_Score + F_Score + M_Score` (range: 3-15)


**K-Means Clustering**

- Outliers capped at the 99th percentile before clustering (configurable in `config.yaml`)
- Features standardised with `StandardScaler` (zero mean, unit variance)
- K=5 chosen via elbow method and silhouette score analysis
- Clusters ranked by median RFM_Score, then median Monetary, to assign business labels


**Future Work**

 1. Short Term

- [ ] Add streaming data support — Accept new transactions incrementally instead of reprocessing the entire dataset from scratch
- [x] Improve edge-case handling — Graceful fallback when dataset has fewer than 5 customers (score bins now scale down automatically)
- [ ] Add segment change tracking — Compare segment assignments across two time periods to identify customers moving between segments

2. Medium Term

- [ ] External database support — Connect directly to a SQL database or data warehouse instead of requiring a static Excel file
- [ ] A/B test integration — Link segment assignments to marketing campaign performance data to measure ROI per segment
- [ ] Predictive churn scoring — Add a binary classifier that predicts which "At Risk" customers are most likely to churn in the next 30/60/90 days
- [ ] Multi-product segmentation — Extend RFM to include product category preferences, enabling segment-specific product recommendations

3. Long Term

- [ ] Real-time dashboard — Replace static CSV loading with live database queries and auto-refresh
- [ ] Automated retraining pipeline — Schedule regular re-clustering as new data arrives, with drift detection to alert when segments shift significantly
- [ ] Customer lifetime value (CLV) prediction — Build a probabilistic model (BG/NBD or Pareto/NBD) to forecast future customer value on top of historical RFM
- [ ] Multi-brand / multi-store support — Extend the schema to handle multiple business units with independent or shared segmentation

