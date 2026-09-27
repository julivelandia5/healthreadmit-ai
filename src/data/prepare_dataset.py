from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"


MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def load_dataset() -> pd.DataFrame:
    """Load the raw diabetes hospitalization dataset."""
    return pd.read_csv(
        DATA_PATH,
        na_values=["?"],
        low_memory=False,
    )


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary early-readmission target."""
    df = df.copy()

    df["target"] = (df["readmitted"] == "<30").astype(int)

    return df


def remove_invalid_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with valid readmission outcomes."""
    valid_targets = {"NO", ">30", "<30"}

    return df[df["readmitted"].isin(valid_targets)].copy()


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split encounters by patient so patients do not cross train/test."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_indices, test_indices = next(
        splitter.split(
            df,
            groups=df["patient_nbr"],
        )
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    return train_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Print basic information about the train/test split."""

    train_patients = set(train_df["patient_nbr"])
    test_patients = set(test_df["patient_nbr"])

    overlapping_patients = train_patients.intersection(test_patients)

    print("\n--- Split summary ---")

    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    print(f"Training patients: {train_df['patient_nbr'].nunique():,}")
    print(f"Test patients: {test_df['patient_nbr'].nunique():,}")

    print(
        f"Overlapping patients between train/test: "
        f"{len(overlapping_patients):,}"
    )

    print("\nTraining target distribution:")
    print(train_df["target"].value_counts())
    print(train_df["target"].value_counts(normalize=True))

    print("\nTest target distribution:")
    print(test_df["target"].value_counts())
    print(test_df["target"].value_counts(normalize=True))


def analyze_medications(df: pd.DataFrame) -> None:
    """Inspect medication categories and their distributions."""

    print("\n--- Medication feature analysis ---")

    for column in MEDICATION_COLUMNS:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))

    print("\n--- Medication category summary ---")

    summary_rows = []

    for column in MEDICATION_COLUMNS:
        counts = df[column].value_counts(dropna=False)

        summary_rows.append(
            {
                "feature": column,
                "No": counts.get("No", 0),
                "Steady": counts.get("Steady", 0),
                "Up": counts.get("Up", 0),
                "Down": counts.get("Down", 0),
                "Missing": df[column].isna().sum(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(summary.to_string(index=False))


def main() -> None:
    print("=" * 60)
    print("HEALTHREADMIT AI - DATASET PREPARATION")
    print("=" * 60)

    print(f"\nDataset: {DATA_PATH}")

    df = load_dataset()

    print(f"\nOriginal rows: {len(df):,}")

    df = remove_invalid_target_rows(df)

    print(f"Rows after target validation: {len(df):,}")

    df = create_binary_target(df)

    print("\n--- Binary target ---")
    print(df["target"].value_counts())
    print(df["target"].value_counts(normalize=True))

    train_df, test_df = split_by_patient(df)

    print_split_summary(train_df, test_df)

    analyze_medications(df)


if __name__ == "__main__":
    main()