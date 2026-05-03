"""
Re-evaluate all trained models with the corrected metric.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import (MLFLOW_TRACKING_URI, DATA_PROCESSED, TARGET_COL,
                    CUSTOMER_ID_COL, RANDOM_STATE, N_SPLITS)
from metrics import amex_metric


def reevaluate_all_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load data
    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    X = df.drop(columns=[CUSTOMER_ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    print(f"Data loaded: {X.shape}")
    
    # Load each champion model and evaluate
    results = {}
    
    model_configs = [
        ("lightgbm", "amex-lgbm-champion", mlflow.lightgbm),
        ("xgboost", "amex-xgboost-champion", mlflow.xgboost),
        ("catboost", "amex-catboost-champion", mlflow.catboost),
    ]
    
    for family, model_name, mlflow_module in model_configs:
        try:
            print(f"\n{'='*60}")
            print(f"Evaluating {family}...")
            print('='*60)
            
            # Get latest version
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"  No registered versions for {model_name}")
                continue
            
            latest = max(versions, key=lambda v: int(v.version))
            model_uri = f"models:/{model_name}/{latest.version}"
            
            print(f"  Loading {model_uri}...")
            model = mlflow_module.load_model(model_uri)
            
            # Predict on full data
            print(f"  Generating predictions...")
            if family == "lightgbm":
                preds = model.predict(X)
            else:
                preds = model.predict_proba(X)[:, 1]
            
            # Compute metrics
            m_score = amex_metric(y.values, preds)
            roc_auc = roc_auc_score(y, preds)
            
            print(f"  M Score (corrected): {m_score:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            
            results[family] = {
                "m_score_corrected": m_score,
                "roc_auc": roc_auc,
                "model_version": latest.version,
            }
            
        except Exception as e:
            print(f"  Error evaluating {family}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL COMPARISON (corrected metric)")
    print('='*60)
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values("m_score_corrected", ascending=False)
    print(df_results.to_string())
    
    if not df_results.empty:
        champion = df_results.index[0]
        print(f"\n🏆 True champion: {champion}")
        print(f"   M Score: {df_results.iloc[0]['m_score_corrected']:.4f}")
    
    return df_results


if __name__ == "__main__":
    reevaluate_all_models()