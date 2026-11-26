
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
RAW_TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_NAME = "MP265_PG_Rent_Model"
EXPERIMENT_NAME = "MP265_PG_Rent"

TARGET = "rent_rs"


def prepare_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    return df.drop(
        columns=[
            TARGET,
            "record_id",
            "date",
            "rent_in_signed_agreement",
        ],
        errors="ignore",
    )


def build_pipeline(X):
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_retraining():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    print("Loading datasets...")

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    later["date"] = pd.to_datetime(later["date"], errors="coerce")
    later = later.sort_values("date").reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    later_check = later.iloc[split_index:].copy()

    combined_train = pd.concat(
        [train, later_train],
        ignore_index=True,
    )

    print("Original clean training rows:", len(train))
    print("Later training rows:", len(later_train))
    print("Combined V2 training rows:", len(combined_train))
    print("Final evaluation rows:", len(later_check))

    X_train = prepare_features(combined_train)
    y_train = combined_train[TARGET]

    X_check = prepare_features(later_check)
    y_check = later_check[TARGET]

    # Load existing V1 champion before any possible promotion.
    v1 = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    v1_predictions = v1.predict(X_check)
    v1_mae = mean_absolute_error(y_check, v1_predictions)

    print("\nV1 evaluation MAE:", round(v1_mae, 2))

    # Train V2.
    print("\nTraining V2...")

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    v2_predictions = pipeline.predict(X_check)
    v2_mae = mean_absolute_error(y_check, v2_predictions)

    print("V2 evaluation MAE:", round(v2_mae, 2))

    improvement = (
        (v1_mae - v2_mae) / v1_mae * 100
        if v1_mae > 0 else 0
    )

    promote = v2_mae <= 0.85 * v1_mae

    print("\n========== V1 vs V2 ==========")
    print(f"V1 MAE: {v1_mae:.2f}")
    print(f"V2 MAE: {v2_mae:.2f}")
    print(f"Improvement: {improvement:.2f}%")
    print("Promotion decision:", "PROMOTE V2" if promote else "KEEP V1")

    raw_hash = sha256_file(RAW_TRAIN_PATH)

    with mlflow.start_run(run_name="V2_Retraining") as run:
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("training_rows", len(combined_train))
        mlflow.log_param("later_training_rows", len(later_train))
        mlflow.log_param("evaluation_rows", len(later_check))
        mlflow.log_param("later_split", "70/30 chronological")
        mlflow.log_param("data_sha256", raw_hash)

        mlflow.log_metric("mae_v1_on_check", v1_mae)
        mlflow.log_metric("mae_v2_on_check", v2_mae)
        mlflow.log_metric("improvement_percent", improvement)

        logged_model = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        model_version = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=MODEL_NAME,
        )

        if promote:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=model_version.version,
            )
            decision = "promote_v2"
        else:
            decision = "keep_v1"

        print("\nV2 registered successfully.")
        print("Run ID:", run.info.run_id)
        print("Registered version:", model_version.version)
        print("Decision:", decision)

    print("\n========== FINAL RESULT ==========")
    print("V1 MAE:", round(v1_mae, 2))
    print("V2 MAE:", round(v2_mae, 2))
    print("Improvement:", f"{improvement:.2f}%")
    print("Decision:", decision)


if __name__ == "__main__":
    run_retraining()
