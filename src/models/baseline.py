from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Allow importing project modules from src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from data.preprocessing import (
    load_dataset,
    create_binary_target,
    split_by_patient,
    prepare_features,
)


def train_baseline_model(
    X_train,
    y_train,
) -> LogisticRegression:
    """Train the baseline logistic regression model."""

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model,
    X_data,
    y_data,
    dataset_name,
) -> None:
    """Evaluate the model on a dataset."""

    y_proba = model.predict_proba(X_data)[:, 1]

    y_pred = (
        y_proba >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_data,
        y_proba,
    )

    average_precision = average_precision_score(
        y_data,
        y_proba,
    )

    print(
        f"\n--- {dataset_name} evaluation ---"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.4f}"
    )

    print("\n--- Classification report ---")

    print(
        classification_report(
            y_data,
            y_pred,
            digits=4,
        )
    )

    print("\n--- Confusion matrix ---")

    print(
        confusion_matrix(
            y_data,
            y_pred,
        )
    )


def evaluate_thresholds(
    model,
    X_data,
    y_data,
    dataset_name,
) -> None:
    """Evaluate metrics across classification thresholds."""

    y_proba = model.predict_proba(
        X_data
    )[:, 1]

    thresholds = [
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    print(
        f"\n--- Threshold analysis: "
        f"{dataset_name} ---"
    )

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'Predicted +':>12}"
    )

    print("-" * 58)

    for threshold in thresholds:

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        precision = precision_score(
            y_data,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_data,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_data,
            y_pred,
            zero_division=0,
        )

        predicted_positive = y_pred.sum()

        print(
            f"{threshold:>10.2f} "
            f"{precision:>12.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{predicted_positive:>12,}"
        )


def main() -> None:
    print("=" * 60)
    print("HEALTHREADMIT AI - BASELINE MODEL")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_dataset()

    df = create_binary_target(df)

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        "\nSplitting data by patient..."
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = split_by_patient(df)

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

    print("\nPreparing features...")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        preprocessor,
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
    )

    print(
        f"Training features: "
        f"{X_train.shape[1]:,}"
    )

    print(
        f"Validation features: "
        f"{X_validation.shape[1]:,}"
    )

    print(
        f"Test features: "
        f"{X_test.shape[1]:,}"
    )

    print(
        "\nTraining logistic regression..."
    )

    model = train_baseline_model(
        X_train,
        y_train,
    )

    print(
        "Model training completed."
    )

    print(
        "\nEvaluating baseline on "
        "validation data..."
    )

    evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    evaluate_thresholds(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    print(
        "\n--- Baseline validation ---"
    )

    print(
        "Patient-level split: YES"
    )

    print(
        "Preprocessor fitted on "
        "training data only: YES"
    )

    print(
        "Validation data transformed "
        "without fitting: YES"
    )

    print(
        "Test data transformed "
        "without fitting: YES"
    )

    print(
        "Test set used for model "
        "selection: NO"
    )

    print(
        "Class weighting: balanced"
    )

    print(
        "\nBaseline model completed "
        "successfully."
    )


if __name__ == "__main__":
    main()