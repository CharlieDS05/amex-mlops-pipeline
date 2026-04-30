import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Permite importar config desde cualquier carpeta
sys.path.append(str(Path(__file__).parent))
from config import DATA_RAW, DATA_PROCESSED, TARGET_COL, CUSTOMER_ID_COL


def load_and_engineer_features():
    """
    Carga el dataset AmEx y construye features agregadas a nivel cliente.
    Soporta tanto Parquet como CSV.
    """
    print("Cargando datos...")

    # Detectar formato disponible
    parquet_path = DATA_RAW / "train.parquet"
    csv_path = DATA_RAW / "train_data.csv"

    if parquet_path.exists():
        train = pd.read_parquet(parquet_path)
        print(f"  Cargado desde Parquet: {parquet_path}")
    elif csv_path.exists():
        train = pd.read_csv(csv_path)
        print(f"  Cargado desde CSV: {csv_path}")
    else:
        raise FileNotFoundError(
            f"No se encontró train.parquet ni train_data.csv en {DATA_RAW}"
        )

    print(f"  Filas raw: {len(train):,}")
    print(f"  Columnas: {len(train.columns)}")

    # Cargar labels
    labels_path = DATA_RAW / "train_labels.csv"
    labels = pd.read_csv(labels_path)
    print(f"  Labels cargados: {len(labels):,} clientes únicos")

    # Identificar columnas de features (excluir ID, target, fecha)
    exclude_cols = [CUSTOMER_ID_COL, TARGET_COL, "S_2"]
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    print(f"  Features a agregar: {len(feature_cols)}")

    # Separar numéricas vs categóricas
    cat_features = [c for c in feature_cols if train[c].dtype == "object"]
    num_features = [c for c in feature_cols if c not in cat_features]
    print(f"  Numéricas: {len(num_features)} | Categóricas: {len(cat_features)}")

    # Agregaciones para numéricas
    print("Agregando features numéricas por cliente...")
    num_agg = train.groupby(CUSTOMER_ID_COL)[num_features].agg(
        ["mean", "std", "min", "max", "last"]
    )
    num_agg.columns = ["_".join(col) for col in num_agg.columns]
    num_agg = num_agg.reset_index()

    # Agregaciones para categóricas (last + nunique)
    if cat_features:
        print("Agregando features categóricas por cliente...")
        cat_agg = train.groupby(CUSTOMER_ID_COL)[cat_features].agg(
            ["last", "nunique"]
        )
        cat_agg.columns = ["_".join(col) for col in cat_agg.columns]
        cat_agg = cat_agg.reset_index()
        df_agg = num_agg.merge(cat_agg, on=CUSTOMER_ID_COL)
    else:
        df_agg = num_agg

    # Features temporales
    if "S_2" in train.columns:
        print("Agregando features temporales...")
        train["S_2"] = pd.to_datetime(train["S_2"])
        temporal = train.groupby(CUSTOMER_ID_COL).agg(
            statement_count=("S_2", "count"),
            months_active=("S_2", lambda x: (x.max() - x.min()).days / 30),
        ).reset_index()
        df_agg = df_agg.merge(temporal, on=CUSTOMER_ID_COL)

    # Merge con labels
    df_final = df_agg.merge(labels, on=CUSTOMER_ID_COL)

    # Convertir categóricas a códigos numéricos para que los modelos funcionen
    for col in df_final.select_dtypes(include=["object"]).columns:
        if col != CUSTOMER_ID_COL:
            df_final[col] = pd.Categorical(df_final[col]).codes

    # Guardar
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "train_features.parquet"
    df_final.to_parquet(output_path, index=False)
    print(f"\nDataset procesado guardado: {output_path}")
    print(f"Shape final: {df_final.shape}")
    print(f"Target rate: {df_final[TARGET_COL].mean():.2%}")

    X = df_final.drop(columns=[CUSTOMER_ID_COL, TARGET_COL])
    y = df_final[TARGET_COL]
    return X, y


if __name__ == "__main__":
    X, y = load_and_engineer_features()
    print(f"\nListo. X: {X.shape}, y: {y.shape}")