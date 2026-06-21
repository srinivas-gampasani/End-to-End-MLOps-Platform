"""
Sample Model Training — Customer Churn Classifier
=====================================================
A representative ML model used to exercise the full MLOps platform:
training → MLflow logging → registry → drift detection → canary deployment.

The model itself (a churn classifier) is intentionally simple — the
platform around it (registry, drift, canary, monitoring) is the product.
"""
import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "tenure_months", "monthly_charges", "total_charges", "num_support_tickets",
    "avg_session_minutes", "days_since_last_login", "contract_length_months",
    "num_products", "satisfaction_score", "payment_delay_days",
]


def generate_synthetic_churn_data(n_samples: int = 5000, seed: int = 42, drift_shift: float = 0.0) -> pd.DataFrame:
    """
    Generate synthetic customer churn data.
    `drift_shift` simulates production data drift (e.g. a marketing campaign
    that shifts the customer base — used to demonstrate drift detection).
    """
    rng = np.random.default_rng(seed)

    tenure = rng.exponential(24, n_samples).clip(0, 72)
    monthly_charges = rng.normal(70 + drift_shift * 15, 25, n_samples).clip(10, 200)
    total_charges = monthly_charges * tenure * rng.uniform(0.9, 1.1, n_samples)
    support_tickets = rng.poisson(2 + drift_shift * 1.5, n_samples)
    session_minutes = rng.normal(35 - drift_shift * 8, 15, n_samples).clip(1, 120)
    days_since_login = rng.exponential(5 + drift_shift * 3, n_samples).clip(0, 90)
    contract_length = rng.choice([1, 12, 24], n_samples, p=[0.4, 0.35, 0.25])
    num_products = rng.poisson(1.8, n_samples).clip(1, 6)
    satisfaction = rng.normal(7 - drift_shift * 1.5, 2, n_samples).clip(1, 10)
    payment_delay = rng.exponential(2 + drift_shift * 2, n_samples).clip(0, 60)

    # Churn probability driven by realistic feature relationships
    # Coefficients tuned so each driver contributes a comparable, modest effect
    # given its natural scale. Signal mean ~ -2.5, std ~ 1.0; intercept set so
    # P(churn) ~ 18-22% at the mean, with noise std well below signal std so
    # the relationship remains learnable (ROC-AUC ~0.85+ for a good classifier).
    churn_logit = (
        1.1
        + 0.30 * support_tickets
        + 0.045 * days_since_login
        - 0.030 * tenure
        - 0.28 * satisfaction
        + 0.040 * payment_delay
        - 0.022 * session_minutes
        + 2.5 * drift_shift
        + rng.normal(0, 0.4, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (rng.uniform(0, 1, n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": monthly_charges, "total_charges": total_charges,
        "num_support_tickets": support_tickets, "avg_session_minutes": session_minutes,
        "days_since_last_login": days_since_login, "contract_length_months": contract_length,
        "num_products": num_products, "satisfaction_score": satisfaction,
        "payment_delay_days": payment_delay, "churned": churned,
    })
    return df


def train_champion_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[RandomForestClassifier, Dict, Dict]:
    """Train the 'champion' model — RandomForest baseline."""
    X = df[FEATURE_NAMES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    params = {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 10, "random_state": random_state}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, params, metrics


def train_challenger_model(df: pd.DataFrame, random_state: int = 43) -> Tuple[GradientBoostingClassifier, Dict, Dict]:
    """Train the 'challenger' model — Gradient Boosting (typically stronger)."""
    X = df[FEATURE_NAMES]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.08, "random_state": random_state}
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, params, metrics
