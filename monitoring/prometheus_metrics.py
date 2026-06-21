"""
Prometheus Metrics Exporter
==============================
Exposes platform metrics for Grafana dashboards:
- Model prediction latency (histogram)
- Request count / error count (counters)
- Drift PSI scores (gauges)
- Canary deployment health (gauges)
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Request metrics
PREDICTION_REQUESTS = Counter(
    "mlops_prediction_requests_total", "Total prediction requests", ["model_name", "model_version", "status"]
)
PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds", "Prediction latency in seconds", ["model_name", "model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

# Drift metrics
FEATURE_DRIFT_PSI = Gauge(
    "mlops_feature_drift_psi", "Population Stability Index per feature", ["feature_name"]
)
DRIFT_SHARE = Gauge(
    "mlops_drift_share", "Share of features showing drift", ["model_name"]
)

# Canary deployment metrics
CANARY_TRAFFIC_PCT = Gauge(
    "mlops_canary_traffic_percent", "Percentage of traffic routed to canary", ["deployment_id"]
)
CANARY_SUCCESS_RATE = Gauge(
    "mlops_canary_success_rate", "Canary success rate", ["deployment_id", "track"]
)

# Model registry metrics
MODEL_VERSIONS_TOTAL = Gauge(
    "mlops_model_versions_total", "Total registered model versions", ["model_name", "stage"]
)


def get_metrics() -> bytes:
    """Return Prometheus-formatted metrics for scraping."""
    return generate_latest()


def record_prediction(model_name: str, model_version: str, latency_seconds: float, success: bool):
    status = "success" if success else "error"
    PREDICTION_REQUESTS.labels(model_name=model_name, model_version=model_version, status=status).inc()
    PREDICTION_LATENCY.labels(model_name=model_name, model_version=model_version).observe(latency_seconds)


def record_drift(feature_name: str, psi: float):
    FEATURE_DRIFT_PSI.labels(feature_name=feature_name).set(psi)


def record_drift_share(model_name: str, share: float):
    DRIFT_SHARE.labels(model_name=model_name).set(share)


def record_canary_metrics(deployment_id: str, traffic_pct: float, champion_success_rate: float, canary_success_rate: float):
    CANARY_TRAFFIC_PCT.labels(deployment_id=deployment_id).set(traffic_pct)
    CANARY_SUCCESS_RATE.labels(deployment_id=deployment_id, track="champion").set(champion_success_rate)
    CANARY_SUCCESS_RATE.labels(deployment_id=deployment_id, track="canary").set(canary_success_rate)
