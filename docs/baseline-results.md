# Baseline Model Results

## 1. Experiment Overview

The first baseline model for HealthReadmit AI was implemented using Logistic Regression.

The purpose of this experiment is to establish a reproducible reference point for subsequent model development and comparison.

The current experimental protocol uses a patient-level train/validation/test split. The training set is used to fit the preprocessing pipeline and train the model, while the validation set is used for model evaluation during development and threshold analysis. The test set remains reserved for the final evaluation after model and threshold selection.

---

## 2. Dataset and Split

The complete dataset contains 101,766 hospital encounters from 71,518 unique patients.

The target variable represents early readmission within 30 days:

- `1`: readmitted within 30 days (`<30`)
- `0`: not readmitted within 30 days (`NO` or `>30`)

The data is split at the patient level using `GroupShuffleSplit` to prevent encounters from the same patient appearing in more than one dataset.

Configuration:

- Test size: 20%
- Validation size: 20% of the remaining data
- Random state: 42
- Grouping variable: `patient_nbr`

Current split results:

| Dataset | Encounters | Early Readmission Rate |
|---|---:|---:|
| Training | 65,146 | 11.25% |
| Validation | 16,467 | 11.40% |
| Test | 20,153 | 10.67% |

Patient-level separation was verified across all three datasets:

- Train/Validation overlapping patients: **0**
- Train/Test overlapping patients: **0**
- Validation/Test overlapping patients: **0**

The test set is not used for model selection or threshold selection.

---

## 3. Preprocessing

The preprocessing pipeline is fitted exclusively on the training dataset.

Numerical features are processed using median imputation.

Categorical features are processed using:

1. Most-frequent imputation.
2. One-hot encoding.
3. `handle_unknown="ignore"` for categories not observed during training.

The current transformed datasets contain:

- Training features: 2,287
- Validation features: 2,287
- Test features: 2,287

The following variables are excluded from the baseline:

- `encounter_id`
- `patient_nbr`
- `readmitted`
- `discharge_disposition_id`
- `weight`

The exclusion of `discharge_disposition_id` is intentional because it represents information associated with the patient's discharge and could introduce temporal leakage in an early-readmission prediction scenario.

`weight` is excluded because 96.86% of its values are missing.

The preprocessing pipeline is fitted only on the training data. Validation and test data are transformed using the fitted preprocessing pipeline without refitting it.

---

## 4. Model Configuration

The baseline model is Logistic Regression.

Configuration:

- Solver: `liblinear`
- Maximum iterations: `1000`
- Class weighting: `balanced`
- Random state: `42`

The `liblinear` solver is used for the binary classification baseline and the model completed training without convergence warnings.

---

## 5. Validation Results

The baseline model was evaluated on the validation set.

The validation set contains 16,467 encounters, including 1,877 positive cases corresponding to early readmission.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.6182 |
| Average Precision | 0.1949 |
| Accuracy | 0.6392 |
| Precision - class 1 | 0.1610 |
| Recall - class 1 | 0.5141 |
| F1 - class 1 | 0.2452 |

The positive class corresponds to early readmission (`<30`).

The classification report at the default threshold of 0.50 was:

```text
              precision    recall  f1-score   support

           0     0.9129    0.6553    0.7630     14590
           1     0.1610    0.5141    0.2452      1877

    accuracy                         0.6392     16467
   macro avg     0.5370    0.5847    0.5041     16467
weighted avg     0.8272    0.6392    0.7039     16467

## 6. Validation Confusion Matrix

At the default classification threshold of 0.50, the confusion matrix on the validation set was:

[[9561 5029]
 [ 912  965]]

 Therefore:
- True Negatives (TN): 9,561
- False Positives (FP): 5,029
- False Negatives (FN): 912
- True Positives (TP): 965
The validation set contained 1,877 positive cases.
The model correctly identified 965 of these cases, resulting in a recall of 0.5141 for early readmission.

## 7. Threshold Analysis
Threshold analysis was performed on the validation set to examine the trade-off between precision and recall.

| Threshold | Precision | Recall | F1 | Predicted Positive |
|---:|---:|---:|---:|---:|
| 0.20 | 0.1172 | 0.9675 | 0.2090 | 15,499 |
| 0.30 | 0.1235 | 0.8940 | 0.2170 | 13,586 |
| 0.40 | 0.1376 | 0.7373 | 0.2320 | 10,055 |
| 0.50 | 0.1610 | 0.5141 | 0.2452 | 5,994 |
| 0.60 | 0.1952 | 0.3005 | 0.2367 | 2,889 |
| 0.70 | 0.2534 | 0.1476 | 0.1865 | 1,093 |
| 0.80 | 0.3343 | 0.0597 | 0.1013 | 335 |

The threshold has not yet been selected as the final operating point.
A systematic threshold-selection criterion will be defined before the final test evaluation. The test set will not be used to choose the threshold.


## 8. Interpretation
The baseline establishes a reference point for subsequent experiments.
The current validation results provide the initial benchmark for future model comparisons:
- ROC-AUC: 0.6182
- Average Precision: 0.1949
- Recall for early readmission: 0.5141
- Precision for early readmission: 0.1610
- F1-score for early readmission: 0.2452
Because the target is imbalanced, accuracy is not considered the primary evaluation metric.
The project therefore prioritizes:
- ROC-AUC
- Average Precision
- Recall for early readmission
- Precision for early readmission
- F1-score for early readmission
The baseline should not be interpreted as a final production model. Its primary purpose is to provide a reproducible benchmark for future models, preprocessing experiments, and threshold-selection analysis.

## 9. Historical Baseline Evaluation
An earlier version of the baseline experiment used a two-way patient-level train/test split and evaluated the model directly on the test set. This experiment predates the introduction of the dedicated validation set and is retained only as part of the project's experimental history.

That historical experiment produced:

| Metric | Historical Result |
|---|---:|
| ROC-AUC | 0.6272 |
| Average Precision | 0.1789 |
| Accuracy | 0.6382 |
| Precision - class 1 | 0.1535 |
| Recall - class 1 | 0.5295 |
| F1 - class 1 | 0.2381 |

These results are retained for experimental traceability but are not used as the current benchmark because the project methodology has since been updated to include a dedicated validation set and a strictly reserved test set.
The historical test evaluation should therefore not be interpreted as the final test performance of the current experimental protocol.

## 10. Reproducibility
The current baseline experiment can be reproduced using:
python src\models\baseline.py
The current experiment uses:
- Patient-level train/validation/test splitting.
- Random state 42.
- Training-only preprocessing.
- Logistic Regression with the liblinear solver.
- Balanced class weighting.
- Validation-based evaluation during model development.
- Test set reserved for final evaluation.
The experiment completed without convergence warnings.

## 11. Next Evaluation Step
Before evaluating the final model on the test set, the project will:
1. Define a reproducible threshold-selection criterion using the validation set.
2. Compare additional Machine Learning models under the same patient-level split.
3. Compare their validation performance using the predefined evaluation metrics.
4. Select the model and threshold according to the documented validation procedure.
5. Evaluate the selected configuration once on the held-out test set.
