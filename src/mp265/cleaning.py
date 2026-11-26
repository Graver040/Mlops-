
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "train_clean.csv"

TARGET = "rent_rs"

TARGET_TYPO_IDS = [
    "R00062", "R00132", "R00194", "R00289",
    "R00371", "R00424", "R00476", "R00508",
    "R00523", "R00573", "R00582", "R00691",
    "R00698", "R00939",
]


def normalize_furnishing(value):
    if pd.isna(value):
        return value

    value = str(value).strip().lower()

    mapping = {
        "ful.": "full",
        "full furnishing": "full",
        "fully furnished": "full",
        "sem.": "semi",
        "semi furnished": "semi",
        "semi-furnished": "semi",
        "unf.": "unfurnished",
        "un furnished": "unfurnished",
        "un-furnished": "unfurnished",
    }

    return mapping.get(value, value)


def clean_training_data():

    if not RAW_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Raw training file not found: {RAW_TRAIN_PATH}"
        )

    df = pd.read_csv(RAW_TRAIN_PATH)

    original_rows = len(df)

    # 1. Remove exact duplicate rows.
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    # 2. Correct building-age unit mix-up.
    east_mask = (
        df["locality"].astype(str).str.strip().str.lower()
        == "sector east"
    )

    age_values = df["building_age_yr"]

    unit_mixup_mask = (
        east_mask
        & (age_values > 40)
        & (age_values % 12 == 0)
    )

    df.loc[unit_mixup_mask, "building_age_yr"] = (
        df.loc[unit_mixup_mask, "building_age_yr"] / 12
    )

    # 3. Correct identified target typos.
    typo_mask = df["record_id"].isin(TARGET_TYPO_IDS)

    found_typo_ids = set(df.loc[typo_mask, "record_id"])
    missing_typo_ids = set(TARGET_TYPO_IDS) - found_typo_ids

    if missing_typo_ids:
        raise ValueError(
            f"Expected typo records not found: {sorted(missing_typo_ids)}"
        )

    df.loc[typo_mask, TARGET] = (
        df.loc[typo_mask, TARGET] / 10
    )

    # 4. Normalize furnishing categories.
    df["furnishing"] = df["furnishing"].apply(
        normalize_furnishing
    )

    # 5. Preserve stuck room-area readings without fabricating values.
    stuck_count = int((df["room_area_sqft"] == 130).sum())

    # 6. No sentinel correction: none were confirmed.

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Data cleaning completed successfully.")
    print(f"Original rows: {original_rows}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Target typos corrected: {int(typo_mask.sum())}")
    print(f"Sector East records checked: {int(east_mask.sum())}")
    print(f"Building-age unit corrections: {int(unit_mixup_mask.sum())}")
    print(f"Room-area readings equal to 130 preserved: {stuck_count}")
    print(
        "Furnishing categories:",
        sorted(df["furnishing"].dropna().unique())
    )
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_training_data()
