import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin display
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from mlflow.models.signature import infer_signature

from config import (MLFLOW_TRACKING_URI, EXPERIMENT_NAME,
                    RANDOM_STATE, N_SPLITS, DATA_PROCESSED, TARGET_COL,
                    CUSTOMER_ID_COL)
from metrics import amex_metric


def load_processed_data():
    """Carga el dataset ya procesado por data_pipeline.py"""
    df = pd.read_parquet(DATA_PROCESSED / "train_features.parquet")
    X = df.drop(columns=[CUSTOMER_ID_COL, TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_baseline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Cargando dataset procesado...")
    X, y = load_processed_data()
    print(f"Shape: {X.shape}, target rate: {y.mean():.2%}")

    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        # Tags (contexto del run)
        mlflow.set_tags({
            "model_family": "linear",
            "stage": "baseline",
            "author": "juan",
            "dataset_version": "v1_mean_std_min_max_last",
        })

        # Parámetros
        params = {
            "model_type": "LogisticRegression",
            "C": 0.01,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "n_splits": N_SPLITS,
            "imputer_strategy": "median",
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)

        # Cross-validation
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
        oof_preds = np.zeros(len(y))
        fold_scores = {"amex_m": [], "roc_auc": [], "pr_auc": []}

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"  Fold {fold + 1}/{N_SPLITS}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=params["C"], max_iter=params["max_iter"],
                    solver=params["solver"],
                    class_weight=params["class_weight"],
                    random_state=RANDOM_STATE,
                )),
            ])

            pipe.fit(X_train, y_train)
            val_preds = pipe.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            m_score = amex_metric(y_val.values, val_preds)
            auc = roc_auc_score(y_val, val_preds)
            pr = average_precision_score(y_val, val_preds)

            fold_scores["amex_m"].append(m_score)
            fold_scores["roc_auc"].append(auc)
            fold_scores["pr_auc"].append(pr)

            mlflow.log_metrics({
                f"fold_{fold}_amex_m_score": m_score,
                f"fold_{fold}_roc_auc": auc,
                f"fold_{fold}_pr_auc": pr,
            })
            print(f"    M={m_score:.4f} | AUC={auc:.4f} | PR-AUC={pr:.4f}")

        # Métricas agregadas
        cv_metrics = {
            "cv_amex_m_score_mean": np.mean(fold_scores["amex_m"]),
            "cv_amex_m_score_std": np.std(fold_scores["amex_m"]),
            "cv_roc_auc_mean": np.mean(fold_scores["roc_auc"]),
            "cv_roc_auc_std": np.std(fold_scores["roc_auc"]),
            "cv_pr_auc_mean": np.mean(fold_scores["pr_auc"]),
            "oof_amex_m_score": amex_metric(y.values, oof_preds),
            "oof_roc_auc": roc_auc_score(y, oof_preds),
        }
        mlflow.log_metrics(cv_metrics)

        print(f"\nCV M Score: {cv_metrics['cv_amex_m_score_mean']:.4f} "
              f"± {cv_metrics['cv_amex_m_score_std']:.4f}")

        # Modelo final
        print("Reentrenando en dataset completo...")
        pipe_final = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=params["C"], max_iter=params["max_iter"],
                solver=params["solver"],
                class_weight=params["class_weight"],
                random_state=RANDOM_STATE,
            )),
        ])
        pipe_final.fit(X, y)

        # Loggear modelo
        signature = infer_signature(
            X.head(10), pipe_final.predict_proba(X.head(10))[:, 1]
        )
        mlflow.sklearn.log_model(
            pipe_final, artifact_path="model",
            signature=signature, input_example=X.head(5),
        )

        # Artefacto: distribución de predicciones
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(oof_preds[y == 0], bins=50, alpha=0.6,
                label="No Default", density=True)
        ax.hist(oof_preds[y == 1], bins=50, alpha=0.6,
                label="Default", density=True)
        ax.set_title("OOF Prediction Distribution — Logistic Regression")
        ax.set_xlabel("Predicted probability")
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, "artifacts/oof_distribution.png")
        plt.close()

        print(f"\nRun completado: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    train_baseline()