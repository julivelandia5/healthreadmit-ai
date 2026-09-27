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

    print("\n--- Missing values analysis ---")

    missing_count = df.isna().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_summary = (
        pd.DataFrame(
            {
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
            }
        )
        .query("missing_count > 0")
        .sort_values("missing_percentage", ascending=False)
    )

    if missing_summary.empty:
        print("No missing values found.")
    else:
        print(missing_summary)

    print("\n--- Missingness severity ---")

    for column, row in missing_summary.iterrows():
        percentage = row["missing_percentage"]

        if percentage >= 90:
            category = "Very high"
        elif percentage >= 50:
            category = "High"
        elif percentage >= 20:
            category = "Moderate"
        elif percentage >= 5:
            category = "Low"
        else:
            category = "Very low"

        print(
            f"{column}: "
            f"{percentage:.2f}% missing "
            f"({category})"
        )

    print("\n--- Target distribution ---")
    print(df["readmitted"].value_counts(dropna=False))

    print("\n--- Target proportions ---")
    print(df["readmitted"].value_counts(normalize=True, dropna=False))

    print("\n--- Identifier analysis ---")

    print(f"Unique encounters: {df['encounter_id'].nunique():,}")
    print(f"Unique patients: {df['patient_nbr'].nunique():,}")

    duplicate_encounters = df["encounter_id"].duplicated().sum()
    print(f"Duplicated encounter IDs: {duplicate_encounters:,}")

    patient_encounter_counts = (
        df.groupby("patient_nbr")["encounter_id"].nunique()
    )

    print(
        f"Patients with multiple encounters: "
        f"{(patient_encounter_counts > 1).sum():,}"
    )

    print(
        f"Maximum encounters for a single patient: "
        f"{patient_encounter_counts.max():,}"
    )

    print("\n--- Duplicate row analysis ---")

    duplicate_rows = df.duplicated().sum()

    print(f"Fully duplicated rows: {duplicate_rows:,}")

    print("\n--- Demographic analysis ---")

    demographic_columns = ["race", "gender", "age"]

    for column in demographic_columns:
        print(f"\n{column} distribution:")
        print(df[column].value_counts(dropna=False))

        print(f"\n{column} missing values:")
        print(df[column].isna().sum())

    print("\n--- Early readmission by demographic group ---")

    binary_target = (df["readmitted"] == "<30").astype(int)

    for column in demographic_columns:
        print(f"\n{column}:")

        demographic_rate = (
            df.assign(early_readmission=binary_target)
            .groupby(
                column,
                dropna=False,
            )["early_readmission"]
            .agg(["count", "mean"])
            .sort_values("mean", ascending=False)
        )

        print(demographic_rate)

    print("\n--- Admission source analysis ---")

    admission_source_analysis = (
        df.assign(early_readmission=binary_target)
        .groupby("admission_source_id")["early_readmission"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
    )

    print(admission_source_analysis)


if __name__ == "__main__":
    main()