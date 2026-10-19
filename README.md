# Customer Cluster

An end-to-end customer segmentation system that groups retail customers into five value tiers using **RFM scoring** and **K-Means clustering**, then presents the results in an interactive Streamlit dashboard. Built for marketing and operations teams who need to tailor campaigns, win-backs and retention plays to different kinds of customers — instead of treating them all the same.

## 🚀 Features

- **RFM scoring:** every customer scored 1–5 on Recency, Frequency and Monetary; the three combine into a 3–15 RFM score
- **K-Means segmentation:** outliers capped at the 99th percentile, features standardised, clusters ranked by value and labelled best-to-worst (Champions → Lost / Churned)
- **Smart data cleaning:** drops guest checkouts and duplicates, removes accounting lines (discounts, fees, samples), and nets cancelled orders off each customer's spend
- **Filter-reactive dashboard:** every card, chart and table recomputes from the customers currently in view — segments, RFM score, recency, lifetime spend
- **Action Playbook:** a sized priority target group per segment with downloadable CSV lists
- **Campaign ROI simulator:** break-even response rate, net profit and ROI under editable assumptions
- **Customer Explorer:** customer lookup with percentile ranks, peer comparison and an interactive recency-vs-spend map
- **Honest diagnostics:** elbow and silhouette sweep shown openly — including the fact that K=2 separates best statistically and K=5 is a business choice

## 🛠️ Tech Stack

- **Dashboard (frontend):** Streamlit, Plotly, custom dark-theme CSS
- **Analytics / ML (backend):** Python 3.10+, pandas, NumPy, scikit-learn (K-Means, StandardScaler, silhouette)
- **Batch charts:** matplotlib
- **Data source:** UCI Online Retail dataset (Excel, read via openpyxl)
- **Configuration:** PyYAML — every tunable lives in `config.yaml`
- **Testing / linting:** pytest, pytest-cov, ruff

## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:

- `Python` 3.10 or higher (3.12 recommended)
- `pip` package manager
- Git installed on your local machine
- The [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) from the UCI ML Repository (~22 MB Excel file)

## ⚙️ Installation & Setup

Follow these steps to get your development environment running:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mohaiminul2/Customer-Cluster.git
   cd Customer-Cluster
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   For development (adds pytest, pytest-cov and ruff):
   ```bash
   pip install -e ".[dev]"
   ```

3. **Add the dataset:**
   Download `Online Retail.xlsx` from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) and place it in the `data/` folder.

4. **Run the segmentation pipeline:**
   ```bash
   python rfm_analysis.py
   ```
   This cleans the raw transactions, computes RFM scores, runs K-Means clustering and writes `data/rfm_scored.csv`, `data/segment_summary.csv`, `data/run_metadata.json` and five dark-themed charts under `charts/`.

5. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

The dashboard opens on the Overview tab. Use the sidebar to filter the entire app — quick-view presets, segments, RFM score, days since last purchase and lifetime spend. Navigate the five tabs:

- **Overview:** auto-generated key takeaways, customers-vs-revenue share, revenue concentration (Pareto) curve, segment profile heatmap and a recency × spend grid
- **Segment Deep Dive:** one segment vs all customers — median comparisons, log-scale distributions, top customers and recommended actions
- **Action Playbook:** sized priority target group per segment, downloadable lists and the campaign ROI simulator
- **Customer Explorer:** customer lookup with percentile ranks, peer comparison, interactive customer map and full table with CSV export
- **Method & Diagnostics:** how segments are built, elbow / silhouette by K, RFM score vs segment, definitions and limits

Re-run the pipeline whenever the source data changes, and keep the codebase healthy with:

```bash
pytest            # 97 tests
ruff check .      # lint
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

## ✉️ Developer

```
Mohaiminul Islam | MBA FIN | BS CSE
Email: Mohaiminul.islam503@gmail.com
```
