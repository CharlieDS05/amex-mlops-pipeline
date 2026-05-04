"""
Register the true champion model in MLflow Model Registry.
Promotes the latest version of the champion model to Production stage.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME


def register_true_champion():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Find the corrected_evaluation runs
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'corrected_evaluation'",
        order_by=["metrics.true_m_score_oof DESC"],
    )
    
    if runs_df.empty:
        print("❌ No corrected_evaluation runs found.")
        return
    
    print("=== Final OOF Comparison (corrected metric) ===\n")
    cols = ['tags.model_family', 'metrics.true_m_score_oof',
            'metrics.roc_auc_oof', 'metrics.pr_auc_oof']
    available = [c for c in cols if c in runs_df.columns]
    display_df = runs_df[available].copy()
    display_df.columns = [c.replace('tags.', '').replace('metrics.', '')
                          for c in display_df.columns]
    print(display_df.to_string(index=False))
    
    # Identify champion
    champion = runs_df.iloc[0]
    champion_family = champion['tags.model_family']
    champion_score = champion['metrics.true_m_score_oof']
    
    print(f"\n🏆 Champion: {champion_family}")
    print(f"   True M Score (OOF): {champion_score:.4f}")
    
    # Map family to registered model name
    model_name_map = {
        'xgboost': 'amex-xgboost-champion',
        'lightgbm': 'amex-lgbm-champion',
        'catboost': 'amex-catboost-champion',
    }
    champion_model_name = model_name_map[champion_family]
    
    # Find latest registered version
    versions = client.search_model_versions(f"name='{champion_model_name}'")
    if not versions:
        print(f"\n❌ No registered versions for {champion_model_name}")
        print("   You may need to register the model from a champion_final run first.")
        return
    
    latest = max(versions, key=lambda v: int(v.version))
    print(f"\n📦 Registered model: {champion_model_name}")
    print(f"   Latest version: {latest.version}")
    print(f"   Current stage: {latest.current_stage}")
    
    # Promote to Production
    print(f"\n🚀 Promoting {champion_model_name} v{latest.version} to Production...")
    
    client.transition_model_version_stage(
        name=champion_model_name,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    
    client.update_model_version(
        name=champion_model_name,
        version=latest.version,
        description=(
            f"Production champion. "
            f"True M Score (OOF, 3-fold CV, corrected metric): {champion_score:.4f}. "
            f"ROC-AUC: {champion['metrics.roc_auc_oof']:.4f}. "
            f"Selected over {len(runs_df)} candidate models."
        ),
    )
    
    # Archive other models
    print("\n📦 Archiving other models...")
    for family, model_name in model_name_map.items():
        if family == champion_family:
            continue
        try:
            other_versions = client.search_model_versions(f"name='{model_name}'")
            if other_versions:
                latest_other = max(other_versions, key=lambda v: int(v.version))
                if latest_other.current_stage != "Archived":
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_other.version,
                        stage="Archived",
                    )
                    print(f"   {model_name} v{latest_other.version} → Archived")
        except Exception as e:
            print(f"   Could not archive {model_name}: {e}")
    
    print(f"\n✅ Champion successfully promoted to Production!")
    print(f"   Model: {champion_model_name}")
    print(f"   Version: {latest.version}")
    print(f"   Stage: Production")
    
    return champion_model_name, latest.version


if __name__ == "__main__":
    register_true_champion()