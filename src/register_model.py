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

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'hyperparameter_search'",
        order_by=["metrics.best_oof_amex_m_score DESC"],
    )

    if runs_df.empty:
        print("No hyperparameter_search runs found.")
        return

    comparison = runs_df[[
        "tags.model_family",
        "metrics.best_oof_amex_m_score",
        "run_id",
    ]].copy()
    comparison.columns = ["model_family", "best_m_score", "run_id"]

    print("\n=== Comparación de modelos ===")
    print(comparison.to_string(index=False))

    champion = comparison.iloc[0]
    print(f"\n🏆 Campeón: {champion['model_family']} "
          f"(M Score: {champion['best_m_score']:.4f})")

    # Promover al stage Production
    model_name = f"amex-{champion['model_family']}-champion"
    versions = client.search_model_versions(f"name='{model_name}'")

    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name=model_name, version=latest.version,
            stage="Production", archive_existing_versions=True,
        )
        client.update_model_version(
            name=model_name, version=latest.version,
            description=(
                f"Champion model — M Score OOF: {champion['best_m_score']:.4f}"
            ),
        )
        print(f"\n{model_name} v{latest.version} → Production")

    return comparison


if __name__ == "__main__":
    compare_and_register_champion()