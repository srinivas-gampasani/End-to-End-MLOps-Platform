"""
Drift Detection Engine
========================
Statistical drift detection for production ML models:
- Population Stability Index (PSI) for feature drift
- Kolmogorov-Smirnov test for distribution shift
- Prediction drift monitoring
- Data quality checks (nulls, ranges, cardinality)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class FeatureDriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    is_drifted: bool
    drift_severity: str
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class DriftReport:
    timestamp: str
    overall_drift_detected: bool
    n_features_drifted: int
    n_features_total: int
    drift_share: float
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: Optional[Dict[str, Any]] = None
    data_quality_issues: List[str] = field(default_factory=list)


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no significant change | 0.1-0.2: moderate | >0.2: significant
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, breakpoints)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(ref_counts.sum(), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(cur_counts.sum(), 1))
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def severity_from_psi(psi: float) -> str:
    if psi < 0.1: return "none"
    if psi < 0.2: return "low"
    if psi < 0.3: return "moderate"
    return "high"


class DriftDetector:
    """Compares reference (training) distribution against current production data."""

    def __init__(self, psi_threshold: float = None, ks_pvalue_threshold: float = None):
        self.psi_threshold = psi_threshold or settings.DRIFT_PSI_THRESHOLD
        self.ks_pvalue_threshold = ks_pvalue_threshold or settings.DRIFT_KS_PVALUE_THRESHOLD

    def detect_feature_drift(self, reference_df, current_df, numeric_features) -> List[FeatureDriftResult]:
        results = []
        for feature in numeric_features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            ref_vals = reference_df[feature].dropna().values.astype(float)
            cur_vals = current_df[feature].dropna().values.astype(float)
            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            psi = calculate_psi(ref_vals, cur_vals)
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
            is_drifted = psi > self.psi_threshold or ks_pval < self.ks_pvalue_threshold

            results.append(FeatureDriftResult(
                feature_name=feature, psi=round(psi, 4),
                ks_statistic=round(float(ks_stat), 4), ks_pvalue=round(float(ks_pval), 4),
                is_drifted=bool(is_drifted), drift_severity=severity_from_psi(psi),
                reference_mean=round(float(ref_vals.mean()), 4), current_mean=round(float(cur_vals.mean()), 4),
                reference_std=round(float(ref_vals.std()), 4), current_std=round(float(cur_vals.std()), 4),
            ))
        return results

    def detect_prediction_drift(self, reference_predictions, current_predictions) -> Dict[str, Any]:
        ref = np.asarray(reference_predictions, dtype=float)
        cur = np.asarray(current_predictions, dtype=float)
        psi = calculate_psi(ref, cur)
        ks_stat, ks_pval = stats.ks_2samp(ref, cur)
        return {
            "psi": round(psi, 4), "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pval), 4),
            "is_drifted": bool(psi > self.psi_threshold or ks_pval < self.ks_pvalue_threshold),
            "reference_mean": round(float(ref.mean()), 4), "current_mean": round(float(cur.mean()), 4),
            "drift_severity": severity_from_psi(psi),
        }

    def check_data_quality(self, df, numeric_features) -> List[str]:
        issues = []
        for col in numeric_features:
            if col not in df.columns:
                continue
            null_pct = df[col].isna().mean()
            if null_pct > 0.05:
                issues.append(f"{col}: {null_pct*100:.1f}% null values (threshold: 5%)")
            finite_vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_vals) and np.isinf(df[col].dropna()).any():
                issues.append(f"{col}: contains infinite values")
            if df[col].nunique() == 1:
                issues.append(f"{col}: constant value (no variance) — possible pipeline bug")
        return issues

    def generate_report(self, reference_df, current_df, numeric_features,
                         reference_predictions=None, current_predictions=None) -> DriftReport:
        feature_results = self.detect_feature_drift(reference_df, current_df, numeric_features)
        n_drifted = sum(1 for r in feature_results if r.is_drifted)
        n_total = len(feature_results)
        drift_share = n_drifted / n_total if n_total > 0 else 0.0

        prediction_drift = None
        if reference_predictions is not None and current_predictions is not None:
            prediction_drift = self.detect_prediction_drift(reference_predictions, current_predictions)

        quality_issues = self.check_data_quality(current_df, numeric_features)
        overall_drift = drift_share > 0.3 or (prediction_drift and prediction_drift["is_drifted"])

        return DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            overall_drift_detected=bool(overall_drift),
            n_features_drifted=n_drifted, n_features_total=n_total,
            drift_share=round(drift_share, 3), feature_results=feature_results,
            prediction_drift=prediction_drift, data_quality_issues=quality_issues,
        )


detector = DriftDetector()
