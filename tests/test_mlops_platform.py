"""
Test Suite — End-to-End MLOps Platform
==========================================
"""
import pytest
import numpy as np
import pandas as pd


class TestModelTraining:
    def test_generate_churn_data(self):
        from models.churn_model import generate_synthetic_churn_data
        df = generate_synthetic_churn_data(n_samples=500)
        assert len(df) == 500
        assert "churned" in df.columns
        assert df["churned"].isin([0, 1]).all()

    def test_churn_data_drift_shift(self):
        from models.churn_model import generate_synthetic_churn_data
        df_normal = generate_synthetic_churn_data(n_samples=1000, drift_shift=0.0, seed=1)
        df_shifted = generate_synthetic_churn_data(n_samples=1000, drift_shift=0.6, seed=1)
        # Shifted data should have different mean monthly charges
        assert abs(df_normal["monthly_charges"].mean() - df_shifted["monthly_charges"].mean()) > 1

    def test_train_champion(self):
        from models.churn_model import generate_synthetic_churn_data, train_champion_model
        df = generate_synthetic_churn_data(n_samples=1000)
        model, params, metrics = train_champion_model(df)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert metrics["n_train"] > 0

    def test_train_challenger(self):
        from models.churn_model import generate_synthetic_churn_data, train_challenger_model
        df = generate_synthetic_churn_data(n_samples=1000)
        model, params, metrics = train_challenger_model(df)
        assert 0 <= metrics["roc_auc"] <= 1

    def test_model_predicts_correct_shape(self):
        from models.churn_model import generate_synthetic_churn_data, train_champion_model, FEATURE_NAMES
        df = generate_synthetic_churn_data(n_samples=500)
        model, _, _ = train_champion_model(df)
        preds = model.predict(df[FEATURE_NAMES])
        assert len(preds) == len(df)
        assert set(preds).issubset({0, 1})


class TestDriftDetection:
    def test_psi_identical_distributions(self):
        from drift.detector import calculate_psi
        np.random.seed(0)
        ref = np.random.normal(50, 10, 1000)
        cur = np.random.normal(50, 10, 1000)
        psi = calculate_psi(ref, cur)
        assert psi < 0.1  # near-identical distributions -> low PSI

    def test_psi_shifted_distribution(self):
        from drift.detector import calculate_psi
        np.random.seed(0)
        ref = np.random.normal(50, 10, 1000)
        cur = np.random.normal(80, 10, 1000)  # large shift
        psi = calculate_psi(ref, cur)
        assert psi > 0.2  # significant shift -> high PSI

    def test_severity_levels(self):
        from drift.detector import severity_from_psi
        assert severity_from_psi(0.05) == "none"
        assert severity_from_psi(0.15) == "low"
        assert severity_from_psi(0.25) == "moderate"
        assert severity_from_psi(0.5) == "high"

    def test_detect_feature_drift_no_shift(self):
        from drift.detector import DriftDetector
        from models.churn_model import generate_synthetic_churn_data, FEATURE_NAMES
        det = DriftDetector()
        ref_df = generate_synthetic_churn_data(n_samples=2000, drift_shift=0.0, seed=1)
        cur_df = generate_synthetic_churn_data(n_samples=500, drift_shift=0.0, seed=2)
        results = det.detect_feature_drift(ref_df, cur_df, FEATURE_NAMES)
        assert len(results) == len(FEATURE_NAMES)
        # With no real shift, most features should not be drifted
        n_drifted = sum(1 for r in results if r.is_drifted)
        assert n_drifted <= 3

    def test_detect_feature_drift_with_shift(self):
        from drift.detector import DriftDetector
        from models.churn_model import generate_synthetic_churn_data, FEATURE_NAMES
        det = DriftDetector()
        ref_df = generate_synthetic_churn_data(n_samples=2000, drift_shift=0.0, seed=1)
        cur_df = generate_synthetic_churn_data(n_samples=500, drift_shift=0.8, seed=2)
        results = det.detect_feature_drift(ref_df, cur_df, FEATURE_NAMES)
        n_drifted = sum(1 for r in results if r.is_drifted)
        assert n_drifted >= 1  # strong shift should trigger at least some drift

    def test_data_quality_checks(self):
        from drift.detector import DriftDetector
        det = DriftDetector()
        df = pd.DataFrame({"a": [1, 2, np.nan, np.nan, np.nan, np.nan, 7, 8, 9, 10], "b": [1] * 10})
        issues = det.check_data_quality(df, ["a", "b"])
        assert any("null" in i for i in issues)
        assert any("constant" in i for i in issues)

    def test_generate_full_report(self):
        from drift.detector import DriftDetector
        from models.churn_model import generate_synthetic_churn_data, FEATURE_NAMES
        det = DriftDetector()
        ref_df = generate_synthetic_churn_data(n_samples=1000, seed=1)
        cur_df = generate_synthetic_churn_data(n_samples=500, drift_shift=0.5, seed=2)
        report = det.generate_report(ref_df, cur_df, FEATURE_NAMES)
        assert report.n_features_total == len(FEATURE_NAMES)
        assert 0 <= report.drift_share <= 1
        assert report.timestamp


class TestCanaryDeployment:
    def test_start_canary(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        d = mgr.start_canary("test_model", "1", "2", traffic_pct=10)
        assert d.status == "canary_active"
        assert d.traffic_pct == 10
        assert len(d.events) == 1

    def test_simulate_traffic_splits_correctly(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        d = mgr.start_canary("test_model", "1", "2", traffic_pct=20)
        mgr.simulate_traffic(d.deployment_id, n_requests=1000, canary_error_rate=0.0, champion_error_rate=0.0)
        total = d.champion_metrics.requests + d.canary_metrics.requests
        assert total == 1000
        # Roughly 20% should go to canary (allow variance)
        canary_pct = d.canary_metrics.requests / total
        assert 0.1 < canary_pct < 0.3

    def test_evaluate_promote_when_healthy(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        d = mgr.start_canary("test_model", "1", "2", traffic_pct=50)
        mgr.simulate_traffic(d.deployment_id, n_requests=200, canary_error_rate=0.0, champion_error_rate=0.02)
        result = mgr.evaluate_canary(d.deployment_id)
        assert result["decision"] in ("promote", "hold")  # healthy canary -> promote or still gathering data

    def test_evaluate_rollback_when_unhealthy(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        d = mgr.start_canary("test_model", "1", "2", traffic_pct=50)
        mgr.simulate_traffic(d.deployment_id, n_requests=200, canary_error_rate=0.5, champion_error_rate=0.02)
        result = mgr.evaluate_canary(d.deployment_id)
        assert result["decision"] == "rollback"
        assert d.status == "rolled_back"

    def test_evaluate_hold_with_insufficient_traffic(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        d = mgr.start_canary("test_model", "1", "2", traffic_pct=5)
        mgr.simulate_traffic(d.deployment_id, n_requests=10, canary_error_rate=0.0, champion_error_rate=0.0)
        result = mgr.evaluate_canary(d.deployment_id)
        assert result["decision"] == "hold"

    def test_get_nonexistent_deployment_raises(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        with pytest.raises(ValueError):
            mgr.simulate_traffic("nonexistent-id", n_requests=10)

    def test_list_deployments(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        mgr.start_canary("model_a", "1", "2")
        mgr.start_canary("model_b", "1", "2")
        deployments = mgr.list_deployments()
        assert len(deployments) == 2

    def test_promoted_deployment_sets_100_traffic(self):
        from deployment.canary import CanaryDeploymentManager
        mgr = CanaryDeploymentManager()
        d = mgr.start_canary("test_model", "1", "2", traffic_pct=50)
        mgr.simulate_traffic(d.deployment_id, n_requests=200, canary_error_rate=0.0, champion_error_rate=0.0)
        result = mgr.evaluate_canary(d.deployment_id)
        if result["decision"] == "promote":
            assert d.traffic_pct == 100


class TestPrometheusMetrics:
    def test_record_prediction(self):
        from monitoring.prometheus_metrics import record_prediction, get_metrics
        record_prediction("test_model", "1", 0.05, True)
        metrics_output = get_metrics().decode()
        assert "mlops_prediction_requests_total" in metrics_output

    def test_record_drift_metric(self):
        from monitoring.prometheus_metrics import record_drift, get_metrics
        record_drift("test_feature", 0.15)
        metrics_output = get_metrics().decode()
        assert "mlops_feature_drift_psi" in metrics_output

    def test_get_metrics_returns_bytes(self):
        from monitoring.prometheus_metrics import get_metrics
        result = get_metrics()
        assert isinstance(result, bytes)


@pytest.mark.asyncio
class TestAPI:
    async def test_health(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.get("/api/health")
            assert r.status_code == 200
            assert r.json()["status"] == "healthy"

    async def test_metrics_endpoint(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.get("/metrics")
            assert r.status_code == 200

    async def test_train_endpoint(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.post("/api/train", json={"n_samples": 500})
            assert r.status_code == 200
            d = r.json()
            assert "champion" in d
            assert "challenger" in d
            assert "roc_auc_improvement_pct" in d

    async def test_drift_check_endpoint(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.post("/api/drift/check", json={"drift_shift": 0.5, "n_samples": 500})
            assert r.status_code == 200
            d = r.json()
            assert "overall_drift_detected" in d
            assert "feature_results" in d

    async def test_canary_full_lifecycle(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r1 = await c.post("/api/canary/start", json={"model_name": "test", "champion_version": "1",
                                                           "canary_version": "2", "traffic_pct": 50})
            assert r1.status_code == 200
            dep_id = r1.json()["deployment_id"]

            r2 = await c.post(f"/api/canary/{dep_id}/simulate",
                               json={"n_requests": 200, "canary_error_rate": 0.0, "champion_error_rate": 0.0})
            assert r2.status_code == 200

            r3 = await c.post(f"/api/canary/{dep_id}/evaluate")
            assert r3.status_code == 200
            assert r3.json()["decision"] in ("promote", "hold", "rollback")

    async def test_canary_not_found(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.get("/api/canary/nonexistent")
            assert r.status_code == 404

    async def test_list_canaries(self):
        from httpx import AsyncClient
        from api.server import app
        async with AsyncClient(app=app, base_url="http://test") as c:
            r = await c.get("/api/canary")
            assert r.status_code == 200
            assert "deployments" in r.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
