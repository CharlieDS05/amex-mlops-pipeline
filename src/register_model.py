"""
Compare champion models from all 3 families and register the best one.
It works with the resumable training pattern where each trial is an
independent MLflow run.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME


def compare_and_register_champion():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return

    print(f"Searching for champion runs in experiment: {EXPERIMENT_NAME}")
    print("=" * 70)

    # Find the champion_final runs from each model family
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'champion_final'",
        order_by=["metrics.best_oof_amex_m_score DESC"],
    )

    if runs_df.empty:
        print("No champion_final runs found.")
        print("    Make sure you've completed training for at least one model.")
        print("    Trying to find best run from individual trials instead...\n")
        return find_best_from_trials(experiment, client)

    # Show comparison table
    available_cols = runs_df.columns.tolist()
    desired_cols = [
        "tags.model_family",
        "metrics.best_oof_amex_m_score",
        "metrics.n_completed_trials",
        "run_id",
    ]
    cols_to_show = [c for c in desired_cols if c in available_cols]
    
    comparison = runs_df[cols_to_show].copy()
    comparison.columns = [c.replace("tags.", "").replace("metrics.", "")
                          for c in comparison.columns]

    print("\n=== Model comparison ===")
    print(comparison.to_string(index=False))

    # Identify champion
    champion = comparison.iloc[0]
    champion_family = champion["model_family"]
    champion_score = champion["best_oof_amex_m_score"]

    print(f"\n🏆 Champion: {champion_family} "
          f"(M Score: {champion_score:.4f})")

    # Promote champion to Production
    model_name = f"amex-{champion_family}-champion"
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(f" No registered versions found for {model_name}")
            return comparison

        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        client.update_model_version(
            name=model_name,
            version=latest.version,
            description=(
                f"Champion model selected from {len(comparison)} candidates. "
                f"M Score OOF: {champion_score:.4f}. "
                f"Trained with Optuna TPE sampler."
            ),
        )
        print(f"\n✅ {model_name} v{latest.version} → Production")
    except Exception as e:
        print(f"\n Couldn't promote to Production: {e}")
        print("    You can do this manually in MLflow UI.")

    return comparison


def find_best_from_trials(experiment, client):
    """Fallback: find best trial across all hyperparameter_search runs."""
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'hyperparameter_search'",
        order_by=["metrics.oof_amex_m_score DESC"],
    )

    if runs_df.empty:
        print("No trial runs found either. Has any training completed?")
        return None

    # Show top 10 trials by family
    cols = ["tags.model_family", "tags.mlflow.runName",
            "metrics.oof_amex_m_score", "run_id"]
    available_cols = [c for c in cols if c in runs_df.columns]
    
    print("\n=== Top 10 trials across all models ===")
    print(runs_df[available_cols].head(10).to_string(index=False))

    # Best per family
    if "tags.model_family" in runs_df.columns:
        print("\n=== Best trial per family ===")
        best_per_family = (runs_df.groupby("tags.model_family")
                                  ["metrics.oof_amex_m_score"]
                                  .max()
                                  .sort_values(ascending=False))
        print(best_per_family.to_string())

    return runs_df


if __name__ == "__main__":
    compare_and_register_champion()