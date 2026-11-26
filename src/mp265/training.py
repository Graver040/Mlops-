
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
RAW_TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "MP265_PG_Rent"

TARGET = "rent_rs"
RANDOM_STATE = 42


# ============================================================
# 2. DATA HASHING
# ============================================================

def calculate_data_sha256():
    """
    Calculate SHA256 of the original, unmodified training CSV.
    """

    if not RAW_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Original training dataset not found: {RAW_TRAIN_PATH}"
        )

    sha256 = hashlib.sha256()

    with open(RAW_TRAIN_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

def load_training_data():
    """
    Load cleaned training data and separate features and target.
    """

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)

    # Convert date into useful numerical features.
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # Remove raw date, record ID, target and leakage column.
    X = df.drop(
        columns=[
            TARGET,
            "record_id",
            "date",
            "rent_in_signed_agreement",
        ],
        errors="ignore",
    )

    y = df[TARGET]

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return X, y


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(X):
    """
    Create preprocessing for numerical and categorical features.
    """

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )

    return preprocessor


# ============================================================
# 5. MODEL TRAINING AND COMPARISON
# ============================================================

def train_and_compare_models():
    """
    Train three regression models, evaluate validation MAE,
    and record experiments in MLflow.
    """

    print("Loading training dataset...")

    X, y = load_training_data()

    data_hash = calculate_data_sha256()

    print("Dataset shape:", X.shape)
    print("Original training data SHA256:", data_hash)

    # Create a fixed training and validation split.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # Define candidate models.
    models = {
        "LinearRegression": LinearRegression(),

        "Ridge": Ridge(
            alpha=1.0
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Configure MLflow.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    print("\nStarting model experiments...\n")

    for model_name, estimator in models.items():

        print(f"Training {model_name}...")

        # Build a fresh preprocessor for each model.
        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name):

            # Train the model.
            pipeline.fit(X_train, y_train)

            # Generate validation predictions.
            predictions = pipeline.predict(X_valid)

            # Calculate MAE.
            mae = mean_absolute_error(
                y_valid,
                predictions
            )

            # Log experiment parameters.
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("data_sha256", data_hash)

            # Log validation metric.
            mlflow.log_metric("mae", float(mae))

            # Save the complete preprocessing + model pipeline.
            # cloudpickle avoids the skops trusted-type issue.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append(
                {
                    "model": model_name,
                    "mae": float(mae),
                }
            )

            print(f"{model_name} MAE: {mae:.2f}")

    # Compare all models.
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="mae",
        ascending=True
    ).reset_index(drop=True)

    print("\n================================")
    print("       MODEL COMPARISON")
    print("================================")

    print(results_df.to_string(index=False))

    best_model = results_df.iloc[0]["model"]
    best_mae = results_df.iloc[0]["mae"]

    print("\nBest model:", best_model)
    print(f"Best validation MAE: {best_mae:.2f}")

    return results_df


# ============================================================
# 6. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_and_compare_models()
