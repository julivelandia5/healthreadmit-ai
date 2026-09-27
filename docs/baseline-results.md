# Baseline Model Results

## 1. Experiment Overview

The first baseline model for HealthReadmit AI was implemented using Logistic Regression.

The purpose of this experiment is to establish a reproducible reference point for subsequent model development and comparison.

The baseline uses the same patient-level train/test split and preprocessing strategy defined in the project documentation.

---

## 2. Dataset and Split

The complete dataset contains 101,766 hospital encounters from 71,518 unique patients.

The target variable represents early readmission within 30 days:

- `1`: readmitted within 30 days (`<30`)
- `0`: not readmitted within 30 days (`NO` or `>30`)

The data was split at the patient level using `GroupShuffleSplit` to prevent encounters from the same patient appearing in both training and test sets.

Configuration:

- Test size: 20%
- Random state: 42
- Grouping variable: `patient_nbr`

Split results:

| Dataset | Encounters | Patients |
|---|---:|---:|
| Training | 81,613 | 57,214 |
| Test | 20,153 | 14,304 |

The number of patients shared between training and test sets was:

**0**

---

## 3. Preprocessing

The preprocessing pipeline was fitted exclusively on the training dataset.

Numerical features were processed using median imputation.

Categorical features were processed using:

1. Most-frequent imputation.
2. One-hot encoding.
3. `handle_unknown="ignore"` for categories not observed during training.

The transformed datasets contained:

- Training features: 2,366
- Test features: 2,366

The following variables were excluded from the baseline:

- `encounter_id`
- `patient_nbr`
- `readmitted`
- `discharge_disposition_id`
- `weight`

The exclusion of `discharge_disposition_id` is intentional because it represents information associated with the patient's discharge and could introduce temporal leakage in an early-readmission prediction scenario.

`weight` was excluded because 96.86% of its values are missing.

---

## 4. Model Configuration

The baseline model is Logistic Regression.

Configuration:

- Solver: `liblinear`
- Maximum iterations: `1000`
- Class weighting: `balanced`
- Random state: `42`

The `liblinear` solver was used for the binary classification baseline and the model completed training without convergence warnings.

---

## 5. Evaluation Results

The model was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.6272 |
| Average Precision | 0.1789 |
| Accuracy | 0.6382 |
| Precision - class 1 | 0.1535 |
| Recall - class 1 | 0.5295 |
| F1 - class 1 | 0.2381 |

The positive class corresponds to early readmission (`<30`).

---

## 6. Confusion Matrix

The resulting confusion matrix was:

```text
[[11723  6279]
 [ 1012  1139]]

 Therefore:

- True Negatives (TN): 11,723
- False Positives (FP): 6,279
- False Negatives (FN): 1,012
- True Positives (TP): 1,139

The test set contained 2,151 positive cases.

The model correctly identified 1,139 of these cases, resulting in a recall of 0.5295 for early readmission.

---

## 7. Interpretation

The baseline establishes a reference point for subsequent experiments.

The ROC-AUC of 0.6272 and Average Precision of 0.1789 provide baseline measurements for evaluating future models under the same data split and preprocessing strategy.

Because the target is imbalanced, accuracy is not considered the primary evaluation metric.

The project therefore prioritizes:

- ROC-AUC
- Average Precision
- Recall for early readmission
- Precision for early readmission
- F1-score for early readmission

The baseline should not be interpreted as a final production model. Its primary purpose is to provide a reproducible benchmark for future models and preprocessing experiments.

---

## 8. Reproducibility

The experiment can be reproduced using:

`python src\models\baseline.py`

The model uses:

- Patient-level splitting.
- Random state 42.
- Training-only preprocessing.
- Logistic Regression with the `liblinear` solver.
- Balanced class weighting.

The experiment completed without convergence warnings.