import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from mlflow.models.signature import infer_signature

from config import (MLFLOW_TRACKING_URI, EXPERIMENT_NAME, RANDOM_STATE,
                    N_SPLITS, N_TRIALS, DATA_PROCESSED, TARGET_COL,
                    CUSTOMER_ID_COL)
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

    with mlflow.start_run(run_name=f"lgbm_trial_{trial.number:03d}",
                          nested=True):
        mlflow.set_tags({
            "trial_number": str(trial.number),
            "model_family": "lightgbm",
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

        oof_m = amex_metric(y.values, oof_preds)
        oof_auc = roc_auc_score(y, oof_preds)
        oof_pr = average_precision_score(y, oof_preds)

        mlflow.log_metrics({
            "oof_amex_m_score": oof_m,
            "oof_roc_auc": oof_auc,
            "oof_pr_auc": oof_pr,
        })

    return oof_m


def train_lgbm_with_optuna():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Cargando dataset...")
    X, y = load_processed_data()
    print(f"Shape: {X.shape}")

    with mlflow.start_run(run_name="lgbm_optuna_search") as parent_run:
        mlflow.set_tags({
            "model_family": "lightgbm",
            "stage": "hyperparameter_search",
            "optimizer": "optuna",
            "n_trials": str(N_TRIALS),
        })
        mlflow.log_params({
            "n_trials": N_TRIALS,
            "n_splits": N_SPLITS,
            "optimization_metric": "amex_m_score",
        })

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            study_name="lgbm_amex",
        )
        study.optimize(
            lambda t: objective(t, X, y),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

        best_trial = study.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metrics({
            "best_oof_amex_m_score": best_trial.value,
            "n_completed_trials": len(study.trials),
        })

        # Visualizaciones de Optuna
        try:
            fig1 = optuna.visualization.matplotlib.plot_param_importances(study)
            mlflow.log_figure(fig1.figure, "artifacts/optuna_importance.png")
            plt.close()

            fig2 = optuna.visualization.matplotlib.plot_optimization_history(study)
            mlflow.log_figure(fig2.figure, "artifacts/optuna_history.png")
            plt.close()
        except Exception as e:
            print(f"Warning: visualizaciones Optuna fallaron: {e}")

        # Reentrenar mejor modelo
        print("\nReentrenando mejor modelo en dataset completo...")
        best_params = best_trial.params.copy()
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

        print(f"\n✅ Mejor M Score: {best_trial.value:.4f}")
        return parent_run.info.run_id


if __name__ == "__main__":
    train_lgbm_with_optuna()