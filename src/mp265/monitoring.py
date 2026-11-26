
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train_clean.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_URI = "models:/MP265_PG_Rent_Model@champion"

NUMERIC_FEATURES = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
]


def prepare_features(df):
    """Apply the same feature preparation used during model training."""

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    X = df.drop(
        columns=[
            "rent_rs",
            "record_id",
            "date",
            "rent_in_signed_agreement",
        ],
        errors="ignore",
    )

    return X


def calculate_psi(reference, current, bins=10):
    """Calculate PSI using quantile bins from the reference data."""

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if reference.empty or current.empty:
        raise ValueError("Cannot calculate PSI on empty feature data.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference.quantile(quantiles).to_numpy())

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(
        reference, bins=edges, include_lowest=True
    )
    current_bins = pd.cut(
        current, bins=edges, include_lowest=True
    )

    ref_pct = (
        reference_bins.value_counts(sort=False, normalize=True)
        .to_numpy()
    )
    cur_pct = (
        current_bins.value_counts(sort=False, normalize=True)
        .to_numpy()
    )

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def run_monitoring():

    mlflow.set_tracking_uri(TRACKING_URI)

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    if "rent_rs" not in later.columns:
        raise ValueError("later.csv must contain the rent_rs target.")

    print("Training dataset:", train.shape)
    print("Later dataset:", later.shape)

    # --------------------------------------------------------
    # 1. PSI monitoring
    # --------------------------------------------------------

    print("\n========== FEATURE DRIFT (PSI) ==========")

    psi_results = {}

    for feature in NUMERIC_FEATURES:
        psi = calculate_psi(
            train[feature],
            later[feature],
            bins=10,
        )

        psi_results[feature] = psi
        print(f"{feature}: {psi:.6f}")

    most_drifted = max(psi_results, key=psi_results.get)

    print("\nMost drifted feature:", most_drifted)
    print("Highest PSI:", psi_results[most_drifted])

    # --------------------------------------------------------
    # 2. Version 1 champion evaluation
    # --------------------------------------------------------

    print("\n========== CHAMPION MODEL EVALUATION ==========")

    model = mlflow.pyfunc.load_model(MODEL_URI)

    X_later = prepare_features(later)
    y_later = later["rent_rs"]

    predictions = model.predict(X_later)

    overall_mae = mean_absolute_error(
        y_later,
        predictions,
    )

    later_results = later[["locality"]].copy()
    later_results["actual"] = y_later.to_numpy()
    later_results["prediction"] = predictions

    group_mae = (
        later_results.groupby("locality")
        .apply(
            lambda group: mean_absolute_error(
                group["actual"],
                group["prediction"],
            ),
            include_groups=False,
        )
    )

    print(f"\nOverall V1 MAE: {overall_mae:.2f}")

    print("\nMAE by locality:")
    print(group_mae.to_string())

    worst_group = group_mae.idxmax()

    print("\nWorst-served locality:", worst_group)
    print("Worst locality MAE:", f"{group_mae.max():.2f}")

    # --------------------------------------------------------
    # 3. Summary
    # --------------------------------------------------------

    print("\n========== MONITORING SUMMARY ==========")
    print("PSI values:", psi_results)
    print("Most drifted feature:", most_drifted)
    print("V1 MAE:", round(overall_mae, 2))
    print("Worst-served locality:", worst_group)

    return {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_v1": float(overall_mae),
        "mae_by_group": group_mae.to_dict(),
        "worst_served_group": str(worst_group),
    }


if __name__ == "__main__":
    run_monitoring()
