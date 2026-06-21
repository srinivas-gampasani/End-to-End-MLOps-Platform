"""
MLOps Platform API
=====================
Endpoints:
  GET  /api/health
  GET  /metrics                          -> Prometheus scrape endpoint
  POST /api/train                        -> Train champion + challenger, log to MLflow
  GET  /api/registry/runs                -> List MLflow runs
  GET  /api/registry/models              -> List registered model versions
  POST /api/drift/check                  -> Run drift detection (reference vs current)
  GET  /api/drift/latest                 -> Latest drift report
  POST /api/canary/start                 -> Start a canary deployment
  POST /api/canary/{id}/simulate         -> Simulate traffic through canary
  POST /api/canary/{id}/evaluate         -> Evaluate canary (promote/rollback/hold)
  GET  /api/canary/{id}                  -> Get deployment status
  GET  /api/canary                       -> List all deployments
"""
import logging
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="End-to-End MLOps Platform",
    description="MLflow registry + drift detection + canary deployments + Prometheus monitoring",
    version="1.0.0",
    docs_url="/docs",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_state = {"reference_df": None, "current_df": None, "champion_model": None, "challenger_model": None,
          "champion_metrics": None, "challenger_metrics": None, "latest_drift_report": None}


class TrainRequest(BaseModel):
    n_samples: int = 5000
    drift_shift: float = 0.0


class DriftCheckRequest(BaseModel):
    drift_shift: float = 0.3
    n_samples: int = 1000


class CanaryStartRequest(BaseModel):
    model_name: str = "churn_classifier"
    champion_version: str = "3"
    canary_version: str = "4"
    traffic_pct: int = 10


class CanarySimulateRequest(BaseModel):
    n_requests: int = 200
    canary_error_rate: float = 0.02
    champion_error_rate: float = 0.02


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "1.0.0", "mlflow_uri": settings.MLFLOW_TRACKING_URI}


@app.get("/metrics")
async def metrics():
    from monitoring.prometheus_metrics import get_metrics
    return Response(content=get_metrics(), media_type="text/plain")


@app.post("/api/train")
async def train_models(req: TrainRequest):
    """Train champion (RandomForest) + challenger (GradientBoosting), log both to MLflow."""
    from models.churn_model import generate_synthetic_churn_data, train_champion_model, train_challenger_model
    from registry.model_registry import registry

    df = generate_synthetic_churn_data(n_samples=req.n_samples, drift_shift=0.0)
    _state["reference_df"] = df

    champion_model, champion_params, champion_metrics = train_champion_model(df)
    challenger_model, challenger_params, challenger_metrics = train_challenger_model(df)

    _state["champion_model"] = champion_model
    _state["challenger_model"] = challenger_model
    _state["champion_metrics"] = champion_metrics
    _state["challenger_metrics"] = challenger_metrics

    champion_run_id = None
    challenger_run_id = None
    try:
        champion_run_id = registry.log_training_run(
            champion_model, "churn_classifier", champion_params, champion_metrics,
            run_name="champion_random_forest", tags={"track": "champion"}, register=True,
        )
        challenger_run_id = registry.log_training_run(
            challenger_model, "churn_classifier", challenger_params, challenger_metrics,
            run_name="challenger_gradient_boosting", tags={"track": "challenger"}, register=True,
        )
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")

    improvement = ((challenger_metrics["roc_auc"] - champion_metrics["roc_auc"]) / champion_metrics["roc_auc"]) * 100

    return {
        "champion": {"params": champion_params, "metrics": champion_metrics, "run_id": champion_run_id},
        "challenger": {"params": challenger_params, "metrics": challenger_metrics, "run_id": challenger_run_id},
        "roc_auc_improvement_pct": round(improvement, 2),
        "n_samples": req.n_samples,
    }


@app.get("/api/registry/runs")
async def list_runs(max_results: int = 20):
    from registry.model_registry import registry
    return {"runs": registry.list_runs(max_results=max_results)}


@app.get("/api/registry/models")
async def list_model_versions(model_name: str = "churn_classifier"):
    from registry.model_registry import registry
    return {"model_name": model_name, "versions": registry.list_model_versions(model_name)}


@app.get("/api/registry/experiments")
async def list_experiments():
    from registry.model_registry import registry
    return {"experiments": registry.list_experiments()}


@app.post("/api/drift/check")
async def check_drift(req: DriftCheckRequest):
    """Run drift detection comparing reference (training) data against simulated production data."""
    from models.churn_model import generate_synthetic_churn_data, FEATURE_NAMES
    from drift.detector import detector

    if _state["reference_df"] is None:
        _state["reference_df"] = generate_synthetic_churn_data(n_samples=5000, drift_shift=0.0)

    current_df = generate_synthetic_churn_data(n_samples=req.n_samples, drift_shift=req.drift_shift, seed=99)
    _state["current_df"] = current_df

    reference_predictions = None
    current_predictions = None
    if _state["champion_model"] is not None:
        reference_predictions = _state["champion_model"].predict_proba(_state["reference_df"][FEATURE_NAMES])[:, 1]
        current_predictions = _state["champion_model"].predict_proba(current_df[FEATURE_NAMES])[:, 1]

    report = detector.generate_report(
        _state["reference_df"], current_df, FEATURE_NAMES,
        reference_predictions=reference_predictions, current_predictions=current_predictions,
    )
    _state["latest_drift_report"] = report

    try:
        from monitoring.prometheus_metrics import record_drift, record_drift_share
        for fr in report.feature_results:
            record_drift(fr.feature_name, fr.psi)
        record_drift_share("churn_classifier", report.drift_share)
    except Exception as e:
        logger.warning(f"Prometheus recording failed: {e}")

    return {
        "timestamp": report.timestamp,
        "overall_drift_detected": report.overall_drift_detected,
        "n_features_drifted": report.n_features_drifted,
        "n_features_total": report.n_features_total,
        "drift_share": report.drift_share,
        "feature_results": [vars(fr) for fr in report.feature_results],
        "prediction_drift": report.prediction_drift,
        "data_quality_issues": report.data_quality_issues,
    }


@app.get("/api/drift/latest")
async def get_latest_drift():
    report = _state["latest_drift_report"]
    if report is None:
        return {"status": "no_report_yet"}
    return {
        "timestamp": report.timestamp,
        "overall_drift_detected": report.overall_drift_detected,
        "n_features_drifted": report.n_features_drifted,
        "n_features_total": report.n_features_total,
        "drift_share": report.drift_share,
        "feature_results": [vars(fr) for fr in report.feature_results],
        "prediction_drift": report.prediction_drift,
        "data_quality_issues": report.data_quality_issues,
    }


@app.post("/api/canary/start")
async def start_canary(req: CanaryStartRequest):
    from deployment.canary import manager
    d = manager.start_canary(req.model_name, req.champion_version, req.canary_version, req.traffic_pct)
    return {
        "deployment_id": d.deployment_id, "model_name": d.model_name,
        "champion_version": d.champion_version, "canary_version": d.canary_version,
        "traffic_pct": d.traffic_pct, "status": d.status, "events": d.events,
    }


@app.post("/api/canary/{deployment_id}/simulate")
async def simulate_canary_traffic(deployment_id: str, req: CanarySimulateRequest):
    from deployment.canary import manager
    try:
        d = manager.simulate_traffic(
            deployment_id, n_requests=req.n_requests,
            canary_error_rate=req.canary_error_rate, champion_error_rate=req.champion_error_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "deployment_id": deployment_id,
        "champion_metrics": vars(d.champion_metrics),
        "canary_metrics": vars(d.canary_metrics),
    }


@app.post("/api/canary/{deployment_id}/evaluate")
async def evaluate_canary(deployment_id: str):
    from deployment.canary import manager
    try:
        result = manager.evaluate_canary(deployment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        from monitoring.prometheus_metrics import record_canary_metrics
        d = manager.get_deployment(deployment_id)
        record_canary_metrics(deployment_id, d.traffic_pct, d.champion_metrics.success_rate, d.canary_metrics.success_rate)
    except Exception as e:
        logger.warning(f"Prometheus recording failed: {e}")

    return result


@app.get("/api/canary/{deployment_id}")
async def get_canary(deployment_id: str):
    from deployment.canary import manager
    d = manager.get_deployment(deployment_id)
    if not d:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {
        "deployment_id": d.deployment_id, "model_name": d.model_name,
        "champion_version": d.champion_version, "canary_version": d.canary_version,
        "traffic_pct": d.traffic_pct, "status": d.status,
        "champion_metrics": vars(d.champion_metrics), "canary_metrics": vars(d.canary_metrics),
        "events": d.events, "decision_reason": d.decision_reason,
    }


@app.get("/api/canary")
async def list_canaries():
    from deployment.canary import manager
    deployments = manager.list_deployments()
    return {
        "deployments": [
            {
                "deployment_id": d.deployment_id, "model_name": d.model_name,
                "champion_version": d.champion_version, "canary_version": d.canary_version,
                "traffic_pct": d.traffic_pct, "status": d.status,
                "canary_success_rate": d.canary_metrics.success_rate,
                "champion_success_rate": d.champion_metrics.success_rate,
            }
            for d in deployments
        ]
    }


ui_path = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_path):
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")
