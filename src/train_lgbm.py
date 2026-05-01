"""
Crash-resistant CatBoost training with Optuna.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import gc
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
#import matplotlib
#matplotlib.use("Agg")
from optuna.storages import RDBStorage
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from mlflow.models.signature import infer_signature

from config import (MLFLOW_TRACKING_URI, EXPERIMENT_NAME, RANDOM_STATE,
                    N_SPLITS, N_TRIALS, DATA_PROCESSED, TARGET_COL,
                    CUSTOMER_ID_COL, OPTUNA_STORAGE_URI)
from metrics import amex_metric, amex_metric_lgbm

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_processed_data():
    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    X = df.drop(columns=[CUSTOMER_ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def objective(trial, X, y):
    params = {
        "objective": "binary",
        "metric": "None",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name=f"lgbm_trial_{trial.number:03d}"):
        mlflow.set_tags({
            "trial_number": str(trial.number),
            "model_family": "lightgbm",
            "stage": "hyperparameter_search",
            "study_name": "lgbm_amex_v1",
        })
        mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=params["n_estimators"],
                valid_sets=[dval], feval=amex_metric_lgbm,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            oof_preds[val_idx] = model.predict(
                X_val, num_iteration=model.best_iteration
            )

            # Memory freeing
            del X_train, X_val, y_train, y_val, dtrain, dval, model
            gc.collect()

        oof_m = amex_metric(y.values, oof_preds)
        oof_auc = roc_auc_score(y, oof_preds)
        oof_pr = average_precision_score(y, oof_preds)

        mlflow.log_metrics({
            "oof_amex_m_score": oof_m,
            "oof_roc_auc": oof_auc,
            "oof_pr_auc": oof_pr,
        })

        #Freeing memory at the end of trial
        del oof_preds
        gc.collect()

    return oof_m


def train_lgbm_resumable():
    """Resume-from-anywhere training."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("Loading dataset...")
    X, y = load_processed_data()
    print(f"Shape: {X.shape}")

    # Persistent Optuna storage 
    storage = RDBStorage(url=OPTUNA_STORAGE_URI)
    
    study = optuna.create_study(
        study_name="lgbm_amex_v1",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        storage=storage,
        load_if_exists=True,  # ← KEY: resumes if study exists
    )
    
    # Progress check
    n_completed = len([t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = N_TRIALS - n_completed
    
    print(f"\nStudy status:")
    print(f"  Completed trials: {n_completed} / {N_TRIALS}")
    print(f"  Remaining: {n_remaining}")
    
    if n_completed > 0:
        print(f"  Best so far: {study.best_value:.4f}")
        print(f"  Resuming...\n")
    else:
        print(f"  Starting fresh\n")
    
    if n_remaining <= 0:
        print("Study already complete!")
        return finalize_champion_model(study, X, y)
    
    # Run only the remaining trials
    study.optimize(
        lambda t: objective(t, X, y),
        n_trials=n_remaining,
        show_progress_bar=True,
        gc_after_trial=True,  # extra memory cleanup
    )
    
    return finalize_champion_model(study, X, y)


def finalize_champion_model(study, X, y):
    """After all trials complete, train and register the best model."""
    print("\n" + "="*60)
    print("Finalizing champion LightGBM model")
    print("="*60)
    
    with mlflow.start_run(run_name="lgbm_champion_final") as run:
        mlflow.set_tags({
            "model_family": "lightgbm",
            "stage": "champion_final",
        })
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({
            "best_oof_amex_m_score": study.best_value,
            "n_completed_trials": len(study.trials),
        })

        # Retrain on full data
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "binary", "metric": "None",
            "verbosity": -1, "random_state": RANDOM_STATE, "n_jobs": -1,
        })
        n_est = best_params.pop("n_estimators")

        dtrain = lgb.Dataset(X, label=y)
        final_model = lgb.train(best_params, dtrain, num_boost_round=n_est,
                                feval=amex_metric_lgbm)

        signature = infer_signature(X.head(5), final_model.predict(X.head(5)))
        mlflow.lightgbm.log_model(
            final_model, artifact_path="model",
            signature=signature,
            registered_model_name="amex-lgbm-champion",
            input_example=X.head(5),
        )

        print(f"\nBest M Score: {study.best_value:.4f}")
        print(f"Champion run ID: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    train_lgbm_resumable()