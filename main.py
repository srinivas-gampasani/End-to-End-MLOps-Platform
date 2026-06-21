#!/usr/bin/env python3
"""
End-to-End MLOps Platform
=============================
Usage:
  python main.py              # Start API server
  python main.py --demo       # Run full lifecycle demo (train -> drift -> canary)
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BANNER = """
+================================================================+
|   End-to-End MLOps Platform                                    |
|   MLflow Registry + Drift Detection + Canary Deploy + Prometheus|
+================================================================+
|  API:   http://localhost:8000                                  |
|  Docs:  http://localhost:8000/docs                             |
|  UI:    http://localhost:8000                                  |
|  Metrics: http://localhost:8000/metrics                        |
+================================================================+
"""


def run_server():
    import uvicorn
    from config import settings
    print(BANNER)
    uvicorn.run("api.server:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)


def demo():
    """Full end-to-end lifecycle demo: train -> register -> drift -> canary -> promote."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    console.print(Panel.fit("[bold blue]MLOps Platform — Full Lifecycle Demo[/]"))

    # 1. Train
    console.print("\n[bold yellow]1. Training Champion (RandomForest) vs Challenger (GradientBoosting)[/]")
    from models.churn_model import generate_synthetic_churn_data, train_champion_model, train_challenger_model
    from registry.model_registry import registry

    df = generate_synthetic_churn_data(n_samples=5000)
    champion, champ_params, champ_metrics = train_champion_model(df)
    challenger, chal_params, chal_metrics = train_challenger_model(df)

    table = Table(title="Training Results")
    table.add_column("Metric"); table.add_column("Champion (RF)"); table.add_column("Challenger (GBM)")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        table.add_row(k, f"{champ_metrics[k]:.4f}", f"{chal_metrics[k]:.4f}")
    console.print(table)

    improvement = (chal_metrics["roc_auc"] - champ_metrics["roc_auc"]) / champ_metrics["roc_auc"] * 100
    console.print(f"[green]Challenger improvement: {improvement:+.2f}% ROC-AUC[/]\n")

    # 2. Register to MLflow
    console.print("[bold yellow]2. Logging to MLflow Registry[/]")
    try:
        champ_run = registry.log_training_run(champion, "churn_classifier", champ_params, champ_metrics,
                                               run_name="champion_demo", tags={"track": "champion"})
        chal_run = registry.log_training_run(challenger, "churn_classifier", chal_params, chal_metrics,
                                              run_name="challenger_demo", tags={"track": "challenger"})
        console.print(f"[green]✓ Champion run: {champ_run}[/]")
        console.print(f"[green]✓ Challenger run: {chal_run}[/]\n")
    except Exception as e:
        console.print(f"[yellow]MLflow logging skipped: {e}[/]\n")

    # 3. Drift detection
    console.print("[bold yellow]3. Drift Detection — Simulating Production Data Shift[/]")
    from drift.detector import detector
    from models.churn_model import FEATURE_NAMES

    current_df = generate_synthetic_churn_data(n_samples=1000, drift_shift=0.4, seed=99)
    ref_preds = champion.predict_proba(df[FEATURE_NAMES])[:, 1]
    cur_preds = champion.predict_proba(current_df[FEATURE_NAMES])[:, 1]

    report = detector.generate_report(df, current_df, FEATURE_NAMES,
                                       reference_predictions=ref_preds, current_predictions=cur_preds)

    console.print(f"Overall drift detected: {'[red]YES[/]' if report.overall_drift_detected else '[green]NO[/]'}")
    console.print(f"Features drifted: {report.n_features_drifted}/{report.n_features_total} ({report.drift_share:.1%})")

    drift_table = Table(title="Feature Drift (PSI)")
    drift_table.add_column("Feature"); drift_table.add_column("PSI"); drift_table.add_column("Severity")
    for fr in sorted(report.feature_results, key=lambda x: x.psi, reverse=True)[:6]:
        color = {"none": "green", "low": "yellow", "moderate": "orange3", "high": "red"}[fr.drift_severity]
        drift_table.add_row(fr.feature_name, f"{fr.psi:.3f}", f"[{color}]{fr.drift_severity}[/]")
    console.print(drift_table)
    console.print()

    # 4. Canary deployment
    console.print("[bold yellow]4. Canary Deployment — Progressive Rollout[/]")
    from deployment.canary import manager

    d = manager.start_canary("churn_classifier", "3", "4", traffic_pct=10)
    console.print(f"Started canary {d.deployment_id}: v3 -> v4 @ 10% traffic")

    manager.simulate_traffic(d.deployment_id, n_requests=300, canary_error_rate=0.015, champion_error_rate=0.02)
    result = manager.evaluate_canary(d.deployment_id)

    console.print(f"\nChampion: {result['champion_metrics']['requests']} requests, "
                  f"{result['champion_metrics']['successes']}/{result['champion_metrics']['requests']} success")
    console.print(f"Canary:   {result['canary_metrics']['requests']} requests, "
                  f"{result['canary_metrics']['successes']}/{result['canary_metrics']['requests']} success")
    console.print(f"\n[bold]Decision: [{'green' if result['decision']=='promote' else 'red'}]{result['decision'].upper()}[/][/]")
    console.print(f"Reason: {result['reason']}")

    console.print("\n[bold green]✓ Full MLOps lifecycle demo complete![/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        run_server()
