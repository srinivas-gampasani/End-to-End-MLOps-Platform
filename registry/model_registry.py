"""
Model Registry & Experiment Tracking
======================================
MLflow-backed model lifecycle management:
- Experiment tracking (params, metrics, artifacts)
- Model versioning and staging (None → Staging → Production → Archived)
- Model lineage and comparison
"""
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import mlflow
from mlflow.tracking import MlflowClient

from config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Wraps MLflow tracking + model registry for the platform."""

    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        try:
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")
        self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    def log_training_run(
        self,
        model,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        register: bool = True,
    ) -> str:
        """Log a training run to MLflow and optionally register the model."""
        run_name = run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)

            try:
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name if register else None)
            except Exception as e:
                logger.warning(f"sklearn model logging failed, using joblib fallback: {e}")
                import joblib
                joblib.dump(model, "/tmp/model.joblib")
                mlflow.log_artifact("/tmp/model.joblib")

            run_id = run.info.run_id
            logger.info(f"Logged run {run_id} for {model_name}")

        return run_id

    def get_latest_version(self, model_name: str, stage: Optional[str] = None) -> Optional[Dict]:
        """Get the latest model version, optionally filtered by stage."""
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")
                versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

            if not versions:
                return None

            v = versions[0]
            return {
                "name": v.name,
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "creation_timestamp": v.creation_timestamp,
            }
        except Exception as e:
            logger.warning(f"get_latest_version failed: {e}")
            return None

    def transition_stage(self, model_name: str, version: str, stage: str, archive_existing: bool = True):
        """Transition a model version to a new stage (Staging/Production/Archived)."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        run = self.client.get_run(run_id)
        return dict(run.data.metrics)

    def list_model_versions(self, model_name: str) -> List[Dict]:
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                }
                for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
            ]
        except Exception as e:
            logger.warning(f"list_model_versions failed: {e}")
            return []

    def compare_models(self, model_name: str, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare metrics between two model versions."""
        try:
            va = self.client.get_model_version(model_name, version_a)
            vb = self.client.get_model_version(model_name, version_b)
            metrics_a = self.get_run_metrics(va.run_id)
            metrics_b = self.get_run_metrics(vb.run_id)

            comparison = {}
            for k in set(metrics_a.keys()) | set(metrics_b.keys()):
                comparison[k] = {
                    "version_a": metrics_a.get(k),
                    "version_b": metrics_b.get(k),
                    "delta": (metrics_b.get(k, 0) - metrics_a.get(k, 0)) if k in metrics_a and k in metrics_b else None,
                }
            return comparison
        except Exception as e:
            logger.warning(f"compare_models failed: {e}")
            return {}

    def list_experiments(self) -> List[Dict]:
        try:
            experiments = self.client.search_experiments()
            return [{"name": e.name, "experiment_id": e.experiment_id} for e in experiments]
        except Exception as e:
            logger.warning(f"list_experiments failed: {e}")
            return []

    def list_runs(self, experiment_name: Optional[str] = None, max_results: int = 20) -> List[Dict]:
        try:
            exp_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
            exp = self.client.get_experiment_by_name(exp_name)
            if not exp:
                return []
            runs = self.client.search_runs([exp.experiment_id], max_results=max_results,
                                            order_by=["start_time DESC"])
            return [
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "params": dict(r.data.params),
                    "metrics": dict(r.data.metrics),
                }
                for r in runs
            ]
        except Exception as e:
            logger.warning(f"list_runs failed: {e}")
            return []


registry = ModelRegistry()
