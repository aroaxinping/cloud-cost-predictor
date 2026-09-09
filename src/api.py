"""FastAPI prediction endpoint for VM cost recommendations."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import (
    assess_risk,
    estimate_savings,
    load_config,
    load_fleet_avg_hourly,
    load_models,
    recommend,
)

app = FastAPI(
    title="Cloud Cost Predictor",
    description="Predict VM utilization and recommend cost actions using XGBoost quantile regression.",
    version="1.0.0",
)

_models = None
_quantiles = None
_config = None


def _get_models():
    global _models, _quantiles, _config
    if _models is None:
        _models, _quantiles = load_models()
        _config = load_config()
    return _models, _quantiles, _config


class VMInput(BaseModel):
    instance: str = Field(description="VM identifier")
    cpu_std: float = Field(ge=0, description="CPU standard deviation")
    cpu_min: float = Field(ge=0, le=100, description="CPU minimum %")
    cpu_median: float = Field(ge=0, le=100, description="CPU median %")
    trend: float = Field(default=0.0, description="CPU trend slope")
    cv: float = Field(ge=0, description="Coefficient of variation")


class VMPrediction(BaseModel):
    instance: str
    pred_low: float
    pred_mid: float
    pred_high: float
    action: str
    risk: str
    monthly_savings: float


class BatchRequest(BaseModel):
    vms: list[VMInput] = Field(min_length=1, max_length=10000)


class BatchResponse(BaseModel):
    predictions: list[VMPrediction]
    fleet_avg_hourly: float


@app.post("/predict", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Score a batch of VMs and return recommendations."""
    try:
        models, quantiles, config = _get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Models not loaded: {e}") from e

    import numpy as np

    features = []
    for vm in request.vms:
        features.append([vm.cpu_std, vm.cpu_min, vm.cpu_median, vm.trend, vm.cv])

    X = np.array(features)
    preds = {q: models[q].predict(X) for q in quantiles}
    avg_hourly = load_fleet_avg_hourly()
    thresholds = config["thresholds"]
    margins = config["risk_margins"]

    predictions = []
    for i, vm in enumerate(request.vms):
        low = float(preds[quantiles[0]][i])
        mid = float(preds[quantiles[1]][i])
        high = float(preds[quantiles[2]][i])
        action = recommend(high, mid, thresholds)
        risk = assess_risk(action, high, margins, thresholds)
        savings = estimate_savings(action, avg_hourly)

        predictions.append(VMPrediction(
            instance=vm.instance,
            pred_low=round(low, 2),
            pred_mid=round(mid, 2),
            pred_high=round(high, 2),
            action=action,
            risk=risk,
            monthly_savings=round(savings, 2),
        ))

    return BatchResponse(predictions=predictions, fleet_avg_hourly=round(avg_hourly, 4))


@app.get("/health")
def health():
    return {"status": "ok"}
