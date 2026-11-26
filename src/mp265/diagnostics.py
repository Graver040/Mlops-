
import pandas as pd

FILES = {
    "train": "data/raw/train.csv",
    "later": "data/raw/later.csv",
    "holdout": "data/raw/holdout_features.csv",
}


def inspect_dataset(name, path):
    df = pd.read_csv(path)

    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNumeric summary:")
    print(df.describe().T)

    print("\nCategorical value counts:")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False))


def investigate_seven_issues():
    train = pd.read_csv("data/raw/train.csv")
    later = pd.read_csv("data/raw/later.csv")
    holdout = pd.read_csv("data/raw/holdout_features.csv")

    print("\n" + "=" * 65)
    print("SEVEN DATA ISSUE INVESTIGATIONS")
    print("=" * 65)

    # 1. Possible target leakage
    print("\n1. POSSIBLE LEAKING COLUMNS")
    for col in train.columns:
        if col not in holdout.columns:
            print(f"\nMissing from holdout: {col}")
            if col != "rent_rs":
                print("Correlation with rent_rs:",
                      train[col].corr(train["rent_rs"])
                      if pd.api.types.is_numeric_dtype(train[col])
                      else "Non-numeric")

    # 2. Unit mix-up: compare numeric distributions by locality
    print("\n2. NUMERIC FEATURES BY LOCALITY")
    numeric_cols = train.select_dtypes(include="number").columns

    for col in numeric_cols:
        if col in ["sharing", "meals_per_day", "rent_rs",
                   "rent_in_signed_agreement"]:
            continue

        print(f"\n{col}")
        print(train.groupby("locality")[col].agg(
            ["count", "min", "mean", "max"]
        ))

    # 3. Sentinel values
    print("\n3. POSSIBLE SENTINEL VALUES")
    for col in numeric_cols:
        counts = train[col].value_counts()
        suspicious = counts[counts.index < 0]
        if not suspicious.empty:
            print(f"\n{col}:")
            print(suspicious)

    # 4. Duplicate rows
    print("\n4. DUPLICATE ROWS")
    print("Exact duplicate rows:", train.duplicated().sum())

    # 5. Possible stuck sensor
    print("\n5. POSSIBLE STUCK FEATURES")
    train["date"] = pd.to_datetime(train["date"])
    ordered = train.sort_values("date")

    for col in numeric_cols:
        if col in ["rent_rs", "rent_in_signed_agreement"]:
            continue

        runs = ordered.groupby(
            ordered[col].ne(ordered[col].shift()).cumsum()
        ).agg(
            value=(col, "first"),
            start_date=("date", "min"),
            end_date=("date", "max"),
            rows=(col, "size")
        )

        runs["duration_days"] = (
            runs["end_date"] - runs["start_date"]
        ).dt.days

        suspicious = runs[
            (runs["duration_days"] >= 14) & (runs["rows"] >= 10)
        ]

        if not suspicious.empty:
            print(f"\n{col}:")
            print(suspicious.to_string(index=False))

    # 6. Target values potentially multiplied by 10
    print("\n6. TARGET VALUE INVESTIGATION")
    print(train["rent_rs"].sort_values(ascending=False).head(30).to_string(index=False))

    print("\nTarget quantiles:")
    print(train["rent_rs"].quantile(
        [0.50, 0.90, 0.95, 0.99, 1.00]
    ))

    print("\nTarget ratio: rent_rs / rent_in_signed_agreement")
    print((train["rent_rs"] / train["rent_in_signed_agreement"])
          .describe())

    # 7. Spelling variants
    print("\n7. CATEGORICAL SPELLING VARIANTS")
    for col in train.select_dtypes(include="object").columns:
        if col in ["record_id", "date"]:
            continue

        print(f"\nTRAIN - {col}")
        print(train[col].value_counts(dropna=False).to_string())

        if col in later.columns:
            print(f"\nLATER - {col}")
            print(later[col].value_counts(dropna=False).to_string())


def investigate_sentinels():
    train = pd.read_csv("data/raw/train.csv")

    print("\nSENTINEL VALUE CHECK")

    candidates = [-9999, -999, -1, 999, 9999]

    for col in train.select_dtypes(include="number").columns:
        for value in candidates:
            count = (train[col] == value).sum()

            if count > 0:
                print(
                    f"Column: {col}, "
                    f"Value: {value}, "
                    f"Count: {count}"
                )

def verify_leakage_and_typos():
    train = pd.read_csv("data/raw/train.csv")

    train["target_ratio"] = (
        train["rent_rs"] / train["rent_in_signed_agreement"]
    )

    print("\nLEAKAGE AND TARGET TYPO VERIFICATION")

    print("\nCorrelation before removing typo records:")
    print(train["rent_in_signed_agreement"].corr(train["rent_rs"]))

    typo_rows = train[train["target_ratio"] >= 5]

    print("\nTarget typo count:", len(typo_rows))
    print(typo_rows[
        ["record_id", "rent_in_signed_agreement", "rent_rs", "target_ratio"]
    ])

    clean_rows = train[train["target_ratio"] < 5]

    print("\nCorrelation after removing typo records:")
    print(
        clean_rows["rent_in_signed_agreement"].corr(clean_rows["rent_rs"])
    )


if __name__ == "__main__":
    for name, path in FILES.items():
        inspect_dataset(name, path)

    investigate_seven_issues()
    investigate_sentinels()
    verify_leakage_and_typos()
