# cloud-cost-predictor

Predicting cloud infrastructure waste from 123K real VMs. XGBoost with asymmetric loss and 95% confidence intervals recommends which VMs to terminate, downsize, or keep.

## Problem

Companies waste 30-50% of their cloud spend on idle or oversized virtual machines. This project analyzes a real fleet of 123,000+ VMs, quantifies the waste, and builds a cost-aware predictive model to identify which machines can be safely downsized or terminated.

## Data

**SAP Cloud Infrastructure Dataset** ([Zenodo, CC BY 4.0](https://zenodo.org/records/13772668))
- 123,357 VMs monitored over 31 days (August 2024)
- CPU and memory utilization at daily granularity
- VM size classification (vCPU and RAM categories)

> Raw data is not included in this repo. Download `sap.zip` from the link above and place it in `data/raw/`.

## Key Findings

- Median CPU utilization: **5%**, half the fleet barely runs
- **95% of VMs** are candidates for right-sizing or termination
- Estimated waste: **$5.9M/month** on an $11M fleet (54%)

## Model

XGBoost quantile regression trained on 25 days of CPU time series to predict next-week utilization.

| Metric | Value |
|---|---|
| Algorithm | XGBoost quantile regression (q=0.10, 0.50, 0.95) |
| Median model loss | Asymmetric (3x penalty on underpredictions) |
| Features | 5 of 11 kept after VIF multicollinearity analysis |
| Median model Test MAE | 1.71% |
| 85% prediction interval coverage | 85.6% |
| Train-test gap | 0.04% (no overfitting) |
| Overfitting control | Early stopping, L2 reg, subsample 0.8 |
| Validation | 80/20 VM-level split (9,411 unseen VMs) |

### Cost-aware design

Underpredicting CPU usage is worse than overpredicting: terminating a VM that's actually needed causes downtime, while keeping an idle VM just wastes money. The model addresses this at three levels:

1. **Asymmetric loss**: the median model penalizes underpredictions 3x more than overpredictions
2. **95% confidence threshold**: recommendations use q=0.95 (not q=0.90), so only 5% chance of underestimating
3. **Risk levels**: each recommendation is tagged safe/moderate/risky based on margin to threshold

| Action | VMs | Savings potential |
|---|---|---|
| Terminate | 16,324 | $1,221/month |
| Downsize | 27,799 | $17,040/month |
| Review | 2,753 | $4,687/month |
| Keep | 177 | - |

## Dashboard

Interactive Streamlit app with four pages: fleet waste breakdown, model explainer (quantile bands, asymmetric loss curve, feature importance), recommendations explorer with filters, and a "Try It" page for uploading your own VM data.

```bash
streamlit run app.py
```

## Explainability

SHAP TreeExplainer on the median model provides per-VM feature attribution. The notebook includes beeswarm and waterfall plots showing why specific VMs get their recommendations.

## Project Structure

```
app.py                <- Streamlit dashboard
data/
  raw/                <- source data (not tracked)
  clean/              <- processed datasets
notebooks/
  01_eda.ipynb
  02_cost_analysis.ipynb
  03_predictive_model.ipynb
src/
  ingest.py           <- stream SAP zip to per-VM summaries
  eda.py              <- classify VMs (zombie/idle/oversized/right-sized/hot)
  pricing.py          <- map to EC2 pricing, estimate waste
  predict.py          <- load models, generate recommendations
scripts/
  export_figures.py   <- generate publication-ready figures
models/               <- trained XGBoost models (.json)
tests/                <- smoke tests for prediction pipeline
reports/
  figures/            <- generated plots
config.yaml           <- model hyperparameters and thresholds
Dockerfile            <- containerized deployment
Makefile              <- setup/train/predict/app targets
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Ingest SAP dataset (requires sap.zip in data/raw/)
python src/ingest.py

# Run fleet analysis
python src/eda.py

# Estimate costs
python src/pricing.py

# Generate predictions from trained models
python src/predict.py

# Or pass a custom CSV
python src/predict.py path/to/vm_utilization_summary.csv output.csv

# Launch the dashboard
streamlit run app.py

# Or use Docker
docker build -t cloud-cost-predictor .
docker run -p 8501:8501 cloud-cost-predictor
```

## A Note on the Data

The SAP dataset is from August 2024. This does not affect the analysis:

- **CPU predictions don't expire.** The model predicts utilization (% CPU), not prices. A VM running at 2% in 2024 would still be idle today. Usage patterns are hardware-agnostic and time-independent.
- **Cost estimates are illustrative.** The $5.9M figure uses a simplified EC2 pricing proxy to dimension the problem, not to produce a real invoice. Actual EC2 prices for established instance families (r5, m5, c5) have remained stable.
- **The value is methodological.** The dataset comes from a peer-reviewed academic source (SAP, CC BY 4.0). The contribution is the pipeline -- asymmetric loss, quantile thresholds, per-VM risk scoring -- not the specific dollar amounts.

## Limitations

- **No temporal trend in inference.** The `trend` feature (slope of daily CPU) is available during training but not in `predict.py`, which lacks the raw time series. It defaults to zero, slightly reducing prediction quality for VMs with strong upward/downward trends.
- **CPU-only recommendations.** Memory utilization data exists in the pipeline but is not used in the decision logic. A VM with low CPU but high memory would be flagged for termination incorrectly.
- **Static dataset, no retraining loop.** The model is trained once on 31 days of data. A production system would need periodic retraining to capture seasonal patterns and fleet changes.
- **Proxy pricing, not real billing.** The cost model maps vCPU/RAM categories to EC2 equivalents. Real cloud bills include storage, networking, reserved instances, and volume discounts that this analysis does not capture.

## Next Steps

- **Memory-aware recommendations** -- incorporate `mem_mean` into the decision thresholds so high-memory VMs are not incorrectly flagged
- **SHAP-based recommendation explanations** -- surface the top-3 features driving each VM's recommendation in the dashboard and CSV output
- **Anomaly detection layer** -- flag VMs with recent CPU spikes before recommending termination, even if their monthly average is low
- **AWS Pricing API integration** -- replace the static proxy with real-time EC2 pricing for more accurate savings estimates

## License

Analysis code: MIT. Dataset: CC BY 4.0 (SAP).
