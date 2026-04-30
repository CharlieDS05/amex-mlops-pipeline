import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (MLFLOW_TRACKING_URI, EXPERIMENT_NAME, RANDOM_STATE,
                    N_SPLITS, N_TRIALS, DATA_PROCESSED, TARGET_COL,
                    CUSTOMER_ID_COL)
from metrics import amex_metric

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_processed_data():
    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    X = df.drop(columns=[CUSTOMER_ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def catboost_objective(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
    }

    with mlflow.start_run(run_name=f"catboost_trial_{trial.number:03d}", nested=True):
        mlflow.set_tag("model_family", "catboost")
        mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                      early_stopping_rounds=50, verbose=False)
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        oof_m = amex_metric(y.values, oof_preds)
        mlflow.log_metrics({
            "oof_amex_m_score": oof_m,
            "oof_roc_auc": roc_auc_score(y, oof_preds),
            "oof_pr_auc": average_precision_score(y, oof_preds),
        })

    return oof_m


def train_catboost_with_optuna():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X, y = load_processed_data()

    with mlflow.start_run(run_name="catboost_optuna_search") as parent_run:
        mlflow.set_tags({
            "model_family": "catboost",
            "stage": "hyperparameter_search",
        })
        mlflow.log_param("n_trials", N_TRIALS)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(lambda t: catboost_objective(t, X, y),
                       n_trials=N_TRIALS, show_progress_bar=True)

        mlflow.log_metrics({
            "best_oof_amex_m_score": study.best_value,
            "n_completed_trials": len(study.trials),
        })

        best_params = study.best_params.copy()
        best_params.update({
            "random_seed": RANDOM_STATE, "verbose": False,
            "loss_function": "Logloss",
        })
        final_model = CatBoostClassifier(**best_params)
        final_model.fit(X, y, verbose=False)

        mlflow.catboost.log_model(
            final_model, artifact_path="model",
            registered_model_name="amex-catboost-champion",
        )

        print(f"\n✅ CatBoost Mejor M Score: {study.best_value:.4f}")
        return parent_run.info.run_id


if __name__ == "__main__":
    train_catboost_with_optuna()