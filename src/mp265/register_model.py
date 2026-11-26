
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

EXPERIMENT_NAME = "MP265_PG_Rent"
REGISTERED_MODEL_NAME = "MP265_PG_Rent_Model"


def register_champion():

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError("MLflow experiment not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.mae ASC"],
        max_results=100,
    )

    if not runs:
        raise ValueError("No successful MLflow runs found.")

    best_run = runs[0]

    if "mae" not in best_run.data.metrics:
        raise ValueError("Best run does not contain MAE.")

    model_uri = f"runs:/{best_run.info.run_id}/model"

    print("Best run:", best_run.info.run_id)
    print("Best model:", best_run.data.params.get("model_name"))
    print("Best MAE:", best_run.data.metrics["mae"])

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print("\nModel registered successfully.")
    print("Registered name:", REGISTERED_MODEL_NAME)
    print("Version:", registered.version)
    print("Alias: champion")


if __name__ == "__main__":
    register_champion()
