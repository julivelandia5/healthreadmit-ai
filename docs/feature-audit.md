# HealthReadmit AI — Feature Audit

## Purpose

This document records the review of candidate predictive features according to their meaning, availability, missingness, and potential risk of target leakage.

The goal is to establish a reproducible feature-selection rationale before model training.

## Prediction Scenario

The initial prediction scenario assumes that early readmission risk is estimated using information available during the hospitalization encounter before the future readmission outcome is known.

## Initial Feature Categories

| Feature | Initial Role | Leakage Risk | Decision |
|---------|--------------|---------------|----------|
| `encounter_id` | Technical identifier | High | Exclude |
| `patient_nbr` | Patient identifier | High | Exclude from model features |
| `race` | Demographic | Low | Review for fairness |
| `gender` | Demographic | Low | Review for fairness |
| `age` | Demographic | Low | Review for fairness |
| `weight` | Clinical | To investigate | Pending |
| `admission_type_id` | Admission information | Low initial leakage risk | Candidate feature |
| `discharge_disposition_id` | Discharge information | High temporal leakage risk | Exclude from baseline pending prediction-time review |
| `admission_source_id` | Admission source information | Low initial leakage risk | Candidate feature |
| `time_in_hospital` | Hospitalization duration | Potential temporal risk | Pending |
| `payer_code` | Administrative | To investigate | Pending |
| `medical_specialty` | Clinical/administrative | To investigate | Pending |
| `num_lab_procedures` | Encounter information | To investigate | Pending |
| `num_procedures` | Encounter information | To investigate | Pending |
| `num_medications` | Encounter information | To investigate | Pending |
| `number_outpatient` | Prior utilization | Potentially valid | Pending |
| `number_emergency` | Prior utilization | Potentially valid | Pending |
| `number_inpatient` | Prior utilization | Potentially valid | Pending |
| `diag_1` | Diagnosis | To investigate | Pending |
| `diag_2` | Diagnosis | To investigate | Pending |
| `diag_3` | Diagnosis | To investigate | Pending |
| `number_diagnoses` | Encounter information | To investigate | Pending |
| `max_glu_serum` | Laboratory result | To investigate | Pending |
| `A1Cresult` | Laboratory result | To investigate | Pending |
| Medication variables | Medication information | To investigate | Candidate features after preprocessing |
| `examide` | Medication information | No variance | Exclude |
| `citoglipton` | Medication information | No variance | Exclude |
| `change` | Medication change | To investigate | Pending |
| `diabetesMed` | Medication indicator | To investigate | Pending |
| `weight` | Clinical/demographic information | 96.86% missing | Exclude from baseline |

### High-Missingness Feature Exclusion

The `weight` feature contains 98,569 missing observations out of 101,766 records (96.86%).

Because the observed coverage is extremely low, `weight` will be excluded from the baseline feature set rather than relying on imputation for a variable with insufficient observed information.

This exclusion is specific to the baseline model. Future experiments may reconsider the feature if a justified missingness-aware representation is developed.

### Feature Exclusion Principle

Features with zero variance in the observed dataset will be excluded from the baseline model because they cannot provide discriminatory information.

Very low-variance features will not be automatically removed solely because their prevalence is low. Their usefulness and representation will be evaluated during preprocessing and model development.

## Mapping-Based Leakage Findings

The `IDS_mapping.csv` file was reviewed to understand the semantic meaning of the admission, discharge, and admission-source identifiers.

`admission_type_id` describes the type of admission, such as emergency, urgent, or elective admission, and is considered potentially available at or near the beginning of the encounter.

`admission_source_id` describes the source from which the patient arrived or was referred and is also considered potentially available during admission.

`discharge_disposition_id` describes the patient's disposition at discharge, including destinations such as home, another hospital, skilled nursing facilities, hospice, and other post-discharge settings.

Because discharge disposition describes information generated at the end of the hospitalization, it presents a high risk of temporal leakage under the initial prediction scenario. It will therefore be excluded from the baseline feature set unless a later prediction scenario explicitly justifies its availability.

## Leakage Review Principles

Features will not be excluded solely because they are correlated with the target.

A feature will be considered for exclusion when:

1. It directly contains or reveals the target outcome.
2. It would only become available after the intended prediction point.
3. It contains information derived from the future outcome.
4. Its use would create an unrealistic prediction scenario.

Final feature-selection decisions will be documented after further investigation of the dataset and the intended deployment scenario.

## Admission Source Leakage Review

The observed `admission_source_id` values were compared with the definitions provided in `IDS_mapping.csv`.

The mapping includes a category described as `Readmission to Same Home Health Agency` (`admission_source_id = 19`). However, this value does not occur in the observed dataset.

Therefore, there is no evidence in the current dataset that this specific category provides direct information about the target outcome.

The remaining admission-source categories primarily describe how or from where the patient entered the healthcare encounter. These variables will remain candidates for the baseline feature set, subject to further preprocessing and validation.

Some admission-source categories have very small sample sizes. Their observed early-readmission rates will therefore not be interpreted independently as evidence of meaningful group differences.

## Medication Feature Analysis

The medication-related variables were reviewed to assess their category distributions and variability.

Most medication variables use four possible categories:

- `No`
- `Steady`
- `Up`
- `Down`

The analysis showed substantial variation across medications. Variables such as `metformin`, `glipizide`, `glyburide`, `pioglitazone`, `rosiglitazone`, and especially `insulin` contain enough observations across multiple categories to remain candidates for modeling.

Several variables have extremely low variability. For example:

- `acetohexamide`: 101,765 `No` and 1 `Steady`.
- `troglitazone`: 101,763 `No` and 3 `Steady`.
- `glimepiride-pioglitazone`: 101,765 `No` and 1 `Steady`.
- `metformin-pioglitazone`: 101,765 `No` and 1 `Steady`.

Two variables are completely constant in the observed dataset:

- `examide`: 101,766 `No`.
- `citoglipton`: 101,766 `No`.

Constant features provide no variation for a predictive model and will therefore be excluded from the baseline feature set.

Other extremely low-variance medication features will be reviewed during feature preprocessing rather than being removed solely based on their names.