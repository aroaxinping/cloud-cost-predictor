# cloud-cost-predictor

Predicting cloud infrastructure waste from real VM fleet data.

## Problem

Companies waste 30-50% of their cloud spend on idle or oversized virtual machines. This project analyzes a real fleet of 123,000+ VMs, quantifies the waste, and builds a predictive model to identify which machines can be safely downsized or terminated.

## Data

**SAP Cloud Infrastructure Dataset** (Zenodo, CC BY 4.0)
- 123,357 VMs monitored over 30 days (August 2024)
- CPU and memory utilization at daily granularity
- VM size classification (vCPU and RAM categories)

## Key Findings

- Median CPU utilization: **5%** — half the fleet barely runs
- **95% of VMs** are candidates for right-sizing or termination
- Estimated waste: **$5.9M/month** on an $11M fleet (54%)
- Per-VM average waste: **$48/month**

## Project Structure

```
├── data/
│   ├── raw/            ← source data (not tracked)
│   └── processed/      ← clean datasets
├── notebooks/          ← EDA and presentation notebooks
├── src/                ← analysis and model code
├── models/             ← trained model artifacts
├── reports/
│   └── figures/        ← generated plots
├── scripts/            ← CLI tools
└── requirements.txt
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
```

## License

Analysis code: MIT. Dataset: CC BY 4.0 (SAP).
