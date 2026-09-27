from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"


TARGET_COLUMN = "target"
GROUP_COLUMN = "patient_nbr"

EXCLUDED_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "discharge_disposition_id",
    "weight",
]

CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "age",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
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
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]

NUMERICAL_COLUMNS = [
    "admission_type_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def load_dataset() -> pd.DataFrame:
    """Load the raw dataset."""

    return pd.read_csv(
        DATA_PATH,
        na_values=["?"],
        low_memory=False,
    )


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary early-readmission target."""

    df = df.copy()

    df[TARGET_COLUMN] = (
        df["readmitted"] == "<30"
    ).astype(int)

    return df


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split encounters into train, validation and test without sharing patients."""

    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_val_indices, test_indices = next(
        test_splitter.split(
            df,
            groups=df[GROUP_COLUMN],
        )
    )

    train_val_df = df.iloc[train_val_indices].copy()
    test_df = df.iloc[test_indices].copy()

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )

    train_indices, validation_indices = next(
        validation_splitter.split(
            train_val_df,
            groups=train_val_df[GROUP_COLUMN],
        )
    )

    train_df = train_val_df.iloc[train_indices].copy()
    validation_df = train_val_df.iloc[validation_indices].copy()

    return (
        train_df,
        validation_df,
        test_df,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def prepare_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """Fit preprocessing on train only and transform train/validation/test."""

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
        and column != TARGET_COLUMN
    ]

    X_train = train_df[feature_columns]
    X_validation = validation_df[feature_columns]
    X_test = test_df[feature_columns]

    y_train = train_df[TARGET_COLUMN]
    y_validation = validation_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )

    return (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        preprocessor,
    )


def main() -> None:
    print("=" * 60)
    print("HEALTHREADMIT AI - PREPROCESSING PIPELINE")
    print("=" * 60)

    print(f"\nDataset: {DATA_PATH}")

    df = load_dataset()
    df = create_binary_target(df)

    print("\n--- Dataset ---")
    print(f"Rows: {len(df):,}")

    (
        train_df,
        validation_df,
        test_df,
    ) = split_by_patient(df)

    print("\n--- Patient-level split ---")

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_df):,}"
    )

    train_patients = set(
        train_df[GROUP_COLUMN]
    )

    validation_patients = set(
        validation_df[GROUP_COLUMN]
    )

    test_patients = set(
        test_df[GROUP_COLUMN]
    )

    train_validation_overlap = (
        train_patients.intersection(
            validation_patients
        )
    )

    train_test_overlap = (
        train_patients.intersection(
            test_patients
        )
    )

    validation_test_overlap = (
        validation_patients.intersection(
            test_patients
        )
    )

    print(
        f"Train/Validation overlapping patients: "
        f"{len(train_validation_overlap):,}"
    )

    print(
        f"Train/Test overlapping patients: "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"Validation/Test overlapping patients: "
        f"{len(validation_test_overlap):,}"
    )

    (
        X_train_transformed,
        X_validation_transformed,
        X_test_transformed,
        y_train,
        y_validation,
        y_test,
        preprocessor,
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
    )

    print("\n--- Feature configuration ---")

    print(
        f"Numerical features: "
        f"{len(NUMERICAL_COLUMNS)}"
    )

    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_COLUMNS)}"
    )

    print(
        f"Excluded features: "
        f"{len(EXCLUDED_COLUMNS)}"
    )

    print("\n--- Transformed data ---")

    print(
        f"Training rows: "
        f"{X_train_transformed.shape[0]:,}"
    )

    print(
        f"Training features: "
        f"{X_train_transformed.shape[1]:,}"
    )

    print(
        f"Validation rows: "
        f"{X_validation_transformed.shape[0]:,}"
    )

    print(
        f"Validation features: "
        f"{X_validation_transformed.shape[1]:,}"
    )

    print(
        f"Test rows: "
        f"{X_test_transformed.shape[0]:,}"
    )

    print(
        f"Test features: "
        f"{X_test_transformed.shape[1]:,}"
    )

    print("\n--- Target distribution ---")

    print("Training:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True))

    print("\nValidation:")
    print(y_validation.value_counts())
    print(y_validation.value_counts(normalize=True))

    print("\nTest:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True))

    print("\n--- Preprocessing validation ---")

    print(
        "Preprocessor fitted on training data only: YES"
    )

    print(
        "Validation data transformed without fitting: YES"
    )

    print(
        "Test data transformed without fitting: YES"
    )

    print(
        "Train/Validation patient overlap: "
        f"{len(train_validation_overlap)}"
    )

    print(
        "Train/Test patient overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        "Validation/Test patient overlap: "
        f"{len(validation_test_overlap)}"
    )


if __name__ == "__main__":
    main()