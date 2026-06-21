"""
Canary Deployment Manager
============================
Simulates a Kubernetes-style canary deployment lifecycle for ML models:
1. Deploy challenger alongside champion (small % traffic)
2. Route a configurable % of traffic to canary
3. Monitor canary success rate / latency / error rate
4. Auto-promote if healthy, auto-rollback if unhealthy
5. Zero-downtime cutover via traffic shifting

In production this would call the Kubernetes API (via the `kubernetes` client)
to update a Deployment/Service traffic split (e.g. Istio VirtualService weights
or native K8s rolling update). Here we simulate the state machine + metrics
collection so the full lifecycle can be demonstrated end-to-end without a
live cluster.
"""
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from config import settings

logger = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    CANARY_ACTIVE = "canary_active"
    PROMOTING = "promoting"
    PROMOTED = "promoted"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class TrafficMetrics:
    requests: int = 0
    successes: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.requests if self.requests else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests else 0.0


@dataclass
class CanaryDeployment:
    deployment_id: str
    model_name: str
    champion_version: str
    canary_version: str
    traffic_pct: int
    status: str
    started_at: str
    updated_at: str
    champion_metrics: TrafficMetrics = field(default_factory=TrafficMetrics)
    canary_metrics: TrafficMetrics = field(default_factory=TrafficMetrics)
    events: List[Dict[str, Any]] = field(default_factory=list)
    decision_reason: Optional[str] = None


class CanaryDeploymentManager:
    """Manages canary deployment lifecycle for ML model serving."""

    def __init__(self):
        self.deployments: Dict[str, CanaryDeployment] = {}

    def start_canary(self, model_name: str, champion_version: str, canary_version: str,
                      traffic_pct: int = None) -> CanaryDeployment:
        traffic_pct = traffic_pct or settings.CANARY_TRAFFIC_PCT
        deployment_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()

        deployment = CanaryDeployment(
            deployment_id=deployment_id, model_name=model_name,
            champion_version=champion_version, canary_version=canary_version,
            traffic_pct=traffic_pct, status=DeploymentStatus.CANARY_ACTIVE,
            started_at=now, updated_at=now,
        )
        deployment.events.append({
            "timestamp": now, "event": "canary_started",
            "detail": f"Routing {traffic_pct}% traffic to {canary_version}",
        })
        self.deployments[deployment_id] = deployment
        logger.info(f"Started canary {deployment_id}: {model_name} {champion_version} -> {canary_version} @ {traffic_pct}%")
        return deployment

    def simulate_traffic(self, deployment_id: str, n_requests: int = 100,
                          canary_error_rate: float = 0.02, champion_error_rate: float = 0.02,
                          canary_latency_ms: float = 45.0, champion_latency_ms: float = 50.0) -> CanaryDeployment:
        d = self.deployments.get(deployment_id)
        if not d:
            raise ValueError(f"Deployment {deployment_id} not found")

        for _ in range(n_requests):
            goes_to_canary = random.random() * 100 < d.traffic_pct
            if goes_to_canary:
                d.canary_metrics.requests += 1
                is_error = random.random() < canary_error_rate
                latency = max(5, random.gauss(canary_latency_ms, canary_latency_ms * 0.15))
                d.canary_metrics.total_latency_ms += latency
                d.canary_metrics.errors += int(is_error)
                d.canary_metrics.successes += int(not is_error)
            else:
                d.champion_metrics.requests += 1
                is_error = random.random() < champion_error_rate
                latency = max(5, random.gauss(champion_latency_ms, champion_latency_ms * 0.15))
                d.champion_metrics.total_latency_ms += latency
                d.champion_metrics.errors += int(is_error)
                d.champion_metrics.successes += int(not is_error)

        d.updated_at = datetime.utcnow().isoformat()
        return d

    def evaluate_canary(self, deployment_id: str) -> Dict[str, Any]:
        """
        Decision logic:
        - Rollback if canary success_rate < threshold OR latency > 1.5x champion
        - Promote if canary has enough traffic AND success_rate >= champion's
        - Otherwise: hold
        """
        d = self.deployments.get(deployment_id)
        if not d:
            raise ValueError(f"Deployment {deployment_id} not found")

        min_requests = 30
        threshold = settings.CANARY_SUCCESS_THRESHOLD

        if d.canary_metrics.requests < min_requests:
            decision = "hold"
            reason = f"Insufficient canary traffic ({d.canary_metrics.requests}/{min_requests} requests)"
        elif d.canary_metrics.success_rate < threshold:
            decision = "rollback"
            reason = f"Canary success rate {d.canary_metrics.success_rate:.2%} below threshold {threshold:.2%}"
        elif d.canary_metrics.avg_latency_ms > d.champion_metrics.avg_latency_ms * 1.5 and d.champion_metrics.requests > 0:
            decision = "rollback"
            reason = (f"Canary latency {d.canary_metrics.avg_latency_ms:.1f}ms exceeds "
                      f"1.5x champion latency {d.champion_metrics.avg_latency_ms:.1f}ms")
        elif d.canary_metrics.success_rate >= d.champion_metrics.success_rate - 0.01:
            decision = "promote"
            reason = f"Canary success rate {d.canary_metrics.success_rate:.2%} >= champion {d.champion_metrics.success_rate:.2%}"
        else:
            decision = "hold"
            reason = "Canary underperforming champion but within tolerance — continue monitoring"

        d.decision_reason = reason
        d.events.append({"timestamp": datetime.utcnow().isoformat(), "event": f"evaluation_{decision}", "detail": reason})

        if decision == "promote":
            self._promote(d)
        elif decision == "rollback":
            self._rollback(d)

        return {
            "deployment_id": deployment_id, "decision": decision, "reason": reason,
            "champion_metrics": vars(d.champion_metrics), "canary_metrics": vars(d.canary_metrics),
            "status": d.status,
        }

    def _promote(self, d: CanaryDeployment):
        d.status = DeploymentStatus.PROMOTED
        d.traffic_pct = 100
        d.events.append({"timestamp": datetime.utcnow().isoformat(), "event": "promoted",
                          "detail": f"{d.canary_version} promoted to 100% traffic — now champion"})
        logger.info(f"Deployment {d.deployment_id}: PROMOTED {d.canary_version}")

    def _rollback(self, d: CanaryDeployment):
        d.status = DeploymentStatus.ROLLED_BACK
        d.traffic_pct = 0
        d.events.append({"timestamp": datetime.utcnow().isoformat(), "event": "rolled_back",
                          "detail": f"{d.canary_version} rolled back — {d.champion_version} remains champion"})
        logger.warning(f"Deployment {d.deployment_id}: ROLLED BACK {d.canary_version}")

    def get_deployment(self, deployment_id: str) -> Optional[CanaryDeployment]:
        return self.deployments.get(deployment_id)

    def list_deployments(self) -> List[CanaryDeployment]:
        return sorted(self.deployments.values(), key=lambda d: d.started_at, reverse=True)


manager = CanaryDeploymentManager()
