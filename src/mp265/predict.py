
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
OUTPUT_PATH = ROOT / "reports" / "holdout_predictions.csv"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_URI = "models:/MP265_PG_Rent_Model@champion"


def prepare_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    return df.drop(
        columns=[
            "record_id",
            "date",
            "rent_in_signed_agreement",
            "rent_rs",
        ],
        errors="ignore",
    )


def generate_predictions():
    mlflow.set_tracking_uri(TRACKING_URI)

    print("Loading holdout dataset...")
    holdout = pd.read_csv(HOLDOUT_PATH)

    if "record_id" not in holdout.columns:
        raise ValueError("Holdout dataset is missing record_id.")

    record_ids = holdout["record_id"].copy()

    print("Holdout shape:", holdout.shape)

    print("\nLoading champion model...")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    X = prepare_features(holdout)

    print("Feature shape:", X.shape)

    predictions = model.predict(X)

    if len(predictions) != len(record_ids):
        raise ValueError("Prediction count does not match holdout rows.")

    if pd.isna(predictions).any():
        raise ValueError("Predictions contain missing values.")

    result = pd.DataFrame({
        "record_id": record_ids,
        "prediction": predictions,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nPredictions generated successfully.")
    print("Rows:", len(result))
    print("Columns:", result.columns.tolist())
    print("Saved to:", OUTPUT_PATH)

    print("\nFirst 5 predictions:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    generate_predictions()
