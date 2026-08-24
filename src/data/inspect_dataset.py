from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"


def main() -> None:
    print("=" * 60)
    print("HEALTHREADMIT AI - DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset: {DATA_PATH}")

    df = pd.read_csv(
    DATA_PATH,
    na_values=["?"],
    low_memory=False,
)

    print("\n--- Dataset shape ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- Column names ---")
    for column in df.columns:
        print(column)

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    print("\n--- Target distribution ---")
    print(df["readmitted"].value_counts(dropna=False))

    print("\n--- Target proportions ---")
    print(df["readmitted"].value_counts(normalize=True, dropna=False))


if __name__ == "__main__":
    main()