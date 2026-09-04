# cloud-cost-predictor

Predicting cloud infrastructure waste from real VM fleet data (SAP Cloud Infrastructure Dataset, 123K VMs, Aug 2024).

## What it does

1. Analyzes VM utilization patterns to identify waste
2. Predicts future utilization per VM with uncertainty (quantile regression)
3. Estimates monthly fleet cost and potential savings
4. Recommends which VMs to resize, shut down, or leave alone

## Data

- **SAP Cloud Infrastructure Dataset** (Zenodo, CC BY 4.0): 123,357 VMs with CPU, memory, disk, and network metrics
- **Cloud pricing**: AWS/Azure public pricing APIs

## Structure

```
src/           — analysis and model code
notebooks/     — EDA and presentation notebooks
data/processed/    — processed datasets (tracked)
data/raw/      — source data (not tracked)
scripts/       — CLI tools
```
