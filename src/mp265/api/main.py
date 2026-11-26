
from datetime import date
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[3]

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_URI = "models:/MP265_PG_Rent_Model@champion"


app = FastAPI(
    title="MP265 PG Room Rent Prediction API",
    description="API for PG room rent prediction in Surat",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    date: date

    distance_to_campus_km: float = Field(
        gt=0,
        description="Distance from campus in kilometres",
    )

    room_area_sqft: float = Field(gt=0)

    sharing: int = Field(ge=1, le=4)

    meals_per_day: int = Field(ge=0)

    building_age_yr: float = Field(ge=0)

    wifi_mbps: int = Field(ge=0, le=550)

    furnishing: Literal["full", "semi", "unfurnished"]

    locality: Literal[
        "Market Side",
        "Sector East",
        "Sector West",
    ]


@app.on_event("startup")
def load_model():
    """Load the current MLflow champion model."""
    mlflow.set_tracking_uri(TRACKING_URI)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_rent(data: PredictionInput):
    try:
        model = getattr(app.state, "model", None)

        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded",
            )

        input_data = pd.DataFrame([data.model_dump()])

        input_data["date"] = pd.to_datetime(input_data["date"])
        input_data["month"] = input_data["date"].dt.month
        input_data["day_of_week"] = input_data["date"].dt.dayofweek

        input_data = input_data.drop(columns=["date"])

        prediction = model.predict(input_data)[0]

        return {
            "predicted_rent_rs": round(float(prediction), 2)
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
