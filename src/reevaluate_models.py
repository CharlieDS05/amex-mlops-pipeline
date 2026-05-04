"""
Re-evaluate champion models with proper cross-validation.
Uses best hyperparameters from MLflow (most reliable source) and runs
fresh 3-fold CV to get unbiased OOF M Scores with the corrected metric.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import gc
import mlflow
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (MLFLOW_TRACKING_URI, EXPERIMENT_NAME, DATA_PROCESSED,
                    TARGET_COL, CUSTOMER_ID_COL, RANDOM_STATE, N_SPLITS)
from metrics import amex_metric, amex_metric_lgbm


def get_best_params_from_mlflow(model_family, param_types):
    """
    Extract best hyperparameters from MLflow runs.
    """
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(f"tags.model_family = '{model_family}' "
                       f"AND tags.stage = 'hyperparameter_search'"),
        order_by=["metrics.oof_amex_m_score DESC"],
        max_results=1,
    )
    
    if runs.empty:
        return None
    
    best_run = runs.iloc[0]
    print(f"  Best trial: {best_run['tags.mlflow.runName']}")
    print(f"  Buggy metric score (training time): "
          f"{best_run['metrics.oof_amex_m_score']:.4f}")
    
    # Extract and convert hyperparameters
    params = {}
    for col in best_run.index:
        if col.startswith('params.'):
            param_name = col.replace('params.', '')
            if param_name in param_types:
                value = best_run[col]
                if pd.isna(value):
                    continue
                params[param_name] = param_types[param_name](float(value))
    
    return params


def evaluate_lgbm_oof(X, y, params):
    """Run 3-fold CV with given LightGBM params."""
    full_params = {
        "objective": "binary", "metric": "None", "verbosity": -1,
        "boosting_type": "gbdt", "random_state": RANDOM_STATE, "n_jobs": -1,
        **params,
    }
    n_estimators = full_params.pop("n_estimators")
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"    Fold {fold + 1}/{N_SPLITS}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            full_params, dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval], feval=amex_metric_lgbm,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        
        del X_train, X_val, y_train, y_val, dtrain, dval, model
        gc.collect()
    
    return oof_preds


def evaluate_xgb_oof(X, y, params):
    """Run 3-fold CV with given XGBoost params."""
    full_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "random_state": RANDOM_STATE, "n_jobs": -1,
        **params,
    }
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"    Fold {fold + 1}/{N_SPLITS}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**full_params, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
    
    return oof_preds


def evaluate_catboost_oof(X, y, params):
    """Run 3-fold CV with given CatBoost params."""
    from catboost import CatBoostClassifier
    
    full_params = {
        "random_seed": RANDOM_STATE, "verbose": False,
        "loss_function": "Logloss", "eval_metric": "AUC",
        "task_type": "CPU", "thread_count": -1,
        **params,
    }
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"    Fold {fold + 1}/{N_SPLITS}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = CatBoostClassifier(**full_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=50, verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()
    
    return oof_preds


def reevaluate_all_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    X = df.drop(columns=[CUSTOMER_ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    print(f"Data loaded: {X.shape}\n")
    
    # Define param types per model
    lgbm_param_types = {
        "num_leaves": int, "max_depth": int, "n_estimators": int,
        "min_child_samples": int,
        "learning_rate": float, "subsample": float, "colsample_bytree": float,
        "reg_alpha": float, "reg_lambda": float,
    }
    xgb_param_types = {
        "max_depth": int, "min_child_weight": int, "n_estimators": int,
        "eta": float, "subsample": float, "colsample_bytree": float,
        "reg_alpha": float, "reg_lambda": float, "gamma": float,
    }
    catboost_param_types = {
        "iterations": int, "depth": int, "border_count": int,
        "learning_rate": float, "l2_leaf_reg": float,
        "bagging_temperature": float, "random_strength": float,
    }
    
    model_configs = [
        ("lightgbm", evaluate_lgbm_oof, lgbm_param_types),
        ("xgboost", evaluate_xgb_oof, xgb_param_types),
        ("catboost", evaluate_catboost_oof, catboost_param_types),
    ]
    
    results = {}
    
    for family, eval_func, param_types in model_configs:
        try:
            print(f"{'='*60}")
            print(f"Re-evaluating {family} with proper 3-fold CV...")
            print('='*60)
            
            best_params = get_best_params_from_mlflow(family, param_types)
            
            if best_params is None:
                print(f"  ⚠️  No trials found in MLflow for {family}")
                continue
            
            print(f"  Hyperparameters loaded ({len(best_params)} params)")
            
            oof_preds = eval_func(X, y, best_params)
            
            m_score = amex_metric(y.values, oof_preds)
            roc_auc = roc_auc_score(y, oof_preds)
            pr_auc = average_precision_score(y, oof_preds)
            
            print(f"\n  ✅ TRUE M Score (OOF): {m_score:.4f}")
            print(f"  ✅ ROC-AUC (OOF): {roc_auc:.4f}")
            print(f"  ✅ PR-AUC (OOF): {pr_auc:.4f}")
            
            results[family] = {
                "true_m_score_oof": m_score,
                "roc_auc_oof": roc_auc,
                "pr_auc_oof": pr_auc,
            }
            
            with mlflow.start_run(run_name=f"{family}_corrected_oof_eval"):
                mlflow.set_tags({
                    "model_family": family,
                    "stage": "corrected_evaluation",
                })
                mlflow.log_metrics({
                    "true_m_score_oof": m_score,
                    "roc_auc_oof": roc_auc,
                    "pr_auc_oof": pr_auc,
                })
            
        except Exception as e:
            print(f"\n  ❌ Error evaluating {family}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("FINAL HONEST COMPARISON (3-fold OOF, corrected metric)")
    print('='*60)
    
    if results:
        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values("true_m_score_oof", ascending=False)
        print(df_results.to_string())
        
        champion = df_results.index[0]
        print(f"\n🏆 TRUE Champion: {champion}")
        print(f"   M Score (OOF): {df_results.iloc[0]['true_m_score_oof']:.4f}")
    
    return results


if __name__ == "__main__":
    reevaluate_all_models()