# HealthReadmit AI — Data Card

## 1. Dataset Identity & Provenance

### Dataset Name

Diabetes 130-US Hospitals for Years 1999–2008

### Source

UCI Machine Learning Repository

### Dataset URL

https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008

### Dataset Version

The dataset was retrieved from the official UCI Machine Learning Repository source on August 24, 2026.

### Source File

`diabetes+130-us+hospitals+for+years+1999-2008.zip`

### SHA-256

`f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a`

### Domain

Healthcare / Hospitalization / Diabetes

### Geographic Context

United States

### Time Period

1999–2008

### Data Provider Context

The dataset contains historical patient encounter records collected from 130 hospitals and integrated delivery networks in the United States.

### Intended Use in HealthReadmit AI

The dataset will be used to investigate whether historical hospitalization information contains patterns associated with early hospital readmission within 30 days.

The dataset will be used exclusively for educational, research, and portfolio purposes within this project.

## 2. Dataset Composition

### Initial Dataset Validation

The first programmatic inspection of the downloaded dataset confirmed:

- 101,766 patient encounters.
- 50 columns in the downloaded CSV file.
- The target variable is `readmitted`.
- The target contains three original categories: `NO`, `>30`, and `<30`.
- Missing values are represented as `?` in the raw dataset and were interpreted as missing values during the initial inspection.

### Initial Target Distribution

| Original Target | Records | Proportion |
|----------------|---------:|-----------:|
| `NO` | 54,864 | 53.91% |
| `>30` | 35,545 | 34.93% |
| `<30` | 11,357 | 11.16% |

The `<30` category represents approximately 11.16% of the encounters and will therefore be treated as the positive class during the initial binary classification formulation.

### Initial Missing Data Findings

The initial inspection identified substantial missingness in several variables:

| Feature | Missing Records |
|---------|----------------:|
| `weight` | 98,569 |
| `max_glu_serum` | 96,420 |
| `A1Cresult` | 84,748 |
| `medical_specialty` | 49,949 |
| `payer_code` | 40,256 |
| `race` | 2,273 |
| `diag_3` | 1,423 |
| `diag_2` | 358 |
| `diag_1` | 21 |

These findings are preliminary and will be investigated further during exploratory data analysis and preprocessing.

### Data Type Considerations

The initial pandas inspection generated a mixed-type warning for `payer_code`.

The final data loading strategy will be reviewed during preprocessing to ensure that categorical variables and missing values are handled consistently.

### Number of Instances

101,766 patient encounters.

### Number of Features

47 features.

### Healthcare Institutions

The data represents encounters from 130 hospitals and integrated delivery networks in the United States.

### Time Coverage

The dataset covers patient encounters recorded between 1999 and 2008.

### Data Types

The dataset contains a combination of:

- Numerical variables.
- Categorical variables.
- Demographic information.
- Hospitalization and utilization information.
- Medication-related information.
- Diagnosis-related information.
- Laboratory and clinical information.

### Target-Related Information

The original dataset includes a readmission outcome that distinguishes between:

- No readmission.
- Readmission after more than 30 days.
- Readmission within 30 days.

The original representation of the target will be inspected directly from the downloaded dataset before preprocessing.

## 3. License & Usage

### License

The dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

### Attribution

The dataset source will be explicitly credited to the UCI Machine Learning Repository in the project documentation.

### Project Usage

The dataset will be used for educational, research, and portfolio purposes within HealthReadmit AI.

The project will preserve the original dataset attribution and will document any transformations or derived datasets generated during the Machine Learning workflow.

### Data Distribution

The raw dataset will not be unnecessarily redistributed through the project repository. Instructions for obtaining the dataset from its official source will be documented to support reproducibility.

### Responsible Use

Use of the dataset within this project does not imply that the resulting models are validated for clinical use or that the data can be used to make decisions about real patients.

## 4. Data Characteristics

The dataset contains information related to patient encounters, hospitalization characteristics, diagnoses, medications, laboratory procedures, and healthcare utilization.

The available variables include numerical and categorical attributes and may require different preprocessing strategies depending on their data type and distribution.

The exact data types, unique values, distributions, and relationships between variables will be verified directly from the downloaded dataset during the exploratory data analysis stage.

The project will avoid making assumptions about the statistical properties of the variables before inspecting the actual data.

---

## 5. Missing Data

The dataset documentation indicates the presence of missing values in several variables.

Missing values will be identified and quantified during the data validation and exploratory analysis stages.

The project will analyze:

- The percentage of missing values per feature.
- The number of records affected by missing values.
- Patterns of missingness.
- Whether missing values are concentrated in particular variables or groups.
- Whether missingness could contain meaningful information.

Missing values will not automatically be removed.

The strategy for handling missing values will be selected based on the characteristics and role of each variable and will be documented as part of the preprocessing pipeline.

---

## 6. Target Variable

The original dataset contains a readmission outcome with three possible categories:

- `NO` — No readmission.
- `>30` — Readmission after more than 30 days.
- `<30` — Readmission within 30 days.

For HealthReadmit AI, the target will initially be transformed into a binary classification variable:

| Target | Meaning |
|--------|---------|
| `1` | Readmission within 30 days |
| `0` | No readmission within 30 days |

The treatment of encounters classified as `>30` will be explicitly documented during preprocessing.

The original target distribution will be inspected before the transformation is applied.

The target transformation will be implemented programmatically to ensure reproducibility.

---

## 7. Sensitive Attributes

The dataset contains demographic attributes that may be relevant to fairness analysis.

Potentially sensitive attributes include:

- Race.
- Gender.
- Age.

These variables will be treated carefully because demographic characteristics can be associated with historical disparities and may influence model behavior.

The project will investigate whether model performance differs across relevant demographic groups.

Sensitive attributes will not automatically be removed from the analysis.

Instead, their role in model development and fairness evaluation will be explicitly documented.

---

## 8. Data Quality & Potential Bias

Several data quality and bias considerations will be investigated throughout the project.

### Identifier Structure

The initial dataset inspection found that all 101,766 encounter records have unique `encounter_id` values.

However, the dataset contains 71,518 unique patients identified by `patient_nbr`.

A total of 16,773 patients have multiple hospital encounters in the dataset, with the maximum number of encounters for a single patient being 40.

This distinction is important because the unit of observation is a hospital encounter rather than a unique patient.

During Machine Learning evaluation, patient-level grouping will be considered when creating training and testing partitions to reduce the risk of information leakage between encounters belonging to the same patient.

### Duplicate Record Analysis

The initial inspection found no duplicated `encounter_id` values and no fully duplicated rows.

The results were:

| Check | Result |
|-------|-------:|
| Unique encounter IDs | 101,766 |
| Duplicated encounter IDs | 0 |
| Fully duplicated rows | 0 |

Therefore, no duplicate records will be removed at this stage based solely on exact row duplication.

The presence of multiple encounters belonging to the same patient will be treated separately because repeated encounters are not necessarily duplicate records.

### Demographic Distribution and Initial Outcome Rates

The initial exploratory analysis examined `race`, `gender`, and `age` as demographic variables.

These variables were analyzed descriptively against the binary early-readmission definition (`readmitted == "<30"`).

#### Race

| Group | Records | Early Readmission Rate |
|-------|--------:|-----------------------:|
| Caucasian | 76,099 | 11.29% |
| AfricanAmerican | 19,210 | 11.22% |
| Hispanic | 2,037 | 10.41% |
| Asian | 641 | 10.14% |
| Other | 1,506 | 9.63% |
| Missing | 2,273 | 8.27% |

#### Gender

| Group | Records | Early Readmission Rate |
|-------|--------:|-----------------------:|
| Female | 54,708 | 11.25% |
| Male | 47,055 | 11.06% |
| Unknown/Invalid | 3 | 0.00% |

The `Unknown/Invalid` gender category contains only three records and should not be interpreted as a statistically meaningful subgroup.

#### Age

The age variable is represented as categorical intervals rather than exact ages.

| Age Group | Records | Early Readmission Rate |
|-----------|--------:|-----------------------:|
| `[20-30)` | 1,657 | 14.24% |
| `[80-90)` | 17,197 | 12.08% |
| `[70-80)` | 26,068 | 11.77% |
| `[30-40)` | 3,775 | 11.23% |
| `[60-70)` | 22,483 | 11.13% |
| `[90-100)` | 2,793 | 11.10% |
| `[40-50)` | 9,685 | 10.60% |
| `[50-60)` | 17,256 | 9.67% |
| `[10-20)` | 691 | 5.79% |
| `[0-10)` | 161 | 1.86% |

These results are descriptive and do not establish causal relationships. Differences between groups may reflect sample composition, underlying clinical characteristics, data collection practices, or other factors.

The demographic analysis will inform the later fairness evaluation. Model-level fairness metrics will be assessed separately after model training and prediction.

### Missing Data Analysis

The initial missingness analysis identified substantial variation across variables.

| Feature | Missing Records | Missing Percentage | Severity |
|---------|----------------:|-------------------:|----------|
| `weight` | 98,569 | 96.86% | Very high |
| `max_glu_serum` | 96,420 | 94.75% | Very high |
| `A1Cresult` | 84,748 | 83.28% | High |
| `medical_specialty` | 49,949 | 49.08% | Moderate |
| `payer_code` | 40,256 | 39.56% | Moderate |
| `race` | 2,273 | 2.23% | Very low |
| `diag_3` | 1,423 | 1.40% | Very low |
| `diag_2` | 358 | 0.35% | Very low |
| `diag_1` | 21 | 0.02% | Very low |

These missingness levels will be considered during preprocessing and feature selection.

High missingness alone will not be treated as sufficient evidence to automatically remove a feature. The meaning of missing values, potential predictive value, and risk of introducing bias will be investigated before final preprocessing decisions are made.

In particular, variables such as `weight`, `max_glu_serum`, and `A1Cresult` require additional investigation because their missingness may reflect data availability or whether a measurement was recorded, rather than random missingness.

### Data Quality

The analysis will investigate:

- Missing values.
- Duplicate records.
- Invalid or inconsistent values.
- Unexpected categories.
- High-cardinality categorical variables.
- Numerical outliers.
- Potentially irrelevant identifiers.
- Inconsistent data representations.

### Historical Bias

Because the dataset represents historical healthcare encounters, it may contain patterns influenced by the healthcare system, patient population, clinical practices, documentation practices, and data collection processes of the period in which the data was collected.

These historical patterns may not represent current healthcare environments.

### Representation Bias

The dataset originates from hospitals and integrated delivery networks in the United States. Therefore, its patient population and healthcare context may not be representative of other countries, healthcare systems, institutions, or populations.

### Measurement Bias

Variables may reflect differences in how information was recorded, coded, or documented rather than differences in the underlying clinical condition.

Potential sources of bias identified during the analysis will be documented and considered when interpreting model performance.

---

## 9. Leakage Considerations

Data leakage is considered a major risk for this project.

A feature will be considered potentially problematic if it contains information that would not have been available at the intended prediction point.

The project will therefore investigate whether variables contain information that may directly or indirectly reveal the target outcome.

Potential leakage risks will be evaluated during:

- Data understanding.
- Feature selection.
- Feature engineering.
- Train-test splitting.
- Model preprocessing.

The final feature set will exclude variables identified as inappropriate predictors because they contain information that would only become available after the outcome being predicted.

All leakage-related decisions will be documented.

---

## 10. Preprocessing Plan

The preprocessing pipeline will be developed after the dataset has been inspected.

The initial preprocessing workflow is expected to include:

1. Data loading and validation.
2. Identification of duplicate records.
3. Data type validation.
4. Missing-value analysis.
5. Target transformation.
6. Identification of potential data leakage.
7. Removal or treatment of inappropriate variables.
8. Encoding of categorical variables.
9. Scaling of numerical variables when required by the selected model.
10. Feature engineering.
11. Train-validation-test separation.
12. Validation that preprocessing steps are fitted only on the appropriate training data.

Preprocessing transformations will be implemented using reproducible Python code rather than manually modifying the dataset.

Where appropriate, scikit-learn pipelines will be used to reduce the risk of inconsistent preprocessing between training and inference.

---

## 11. Ethical & Privacy Considerations

The dataset is publicly available and will be used within the scope described by its source and license.

The project will not intentionally collect, request, or process personally identifiable information from real patients.

The application developed for this project will not require users to enter real patient information.

The project will avoid presenting individual predictions as medical recommendations.

Model outputs will be presented as analytical results from a research and educational prototype.

The project will explicitly communicate uncertainty, limitations, potential bias, and the non-clinical nature of the system.

---

## 12. Limitations

The dataset has several characteristics that limit the generalizability of the resulting models.

### Historical Data

The dataset covers hospital encounters from 1999 to 2008. Healthcare practices, technologies, treatments, patient populations, and data collection processes may have changed significantly since that period.

### Geographic Scope

The data originates from healthcare institutions in the United States and may not generalize to healthcare systems in other countries.

### Institutional Scope

Although the dataset represents multiple hospitals and integrated delivery networks, it does not represent every type of healthcare institution or patient population.

### Population Representation

The characteristics of the population represented in the dataset may differ from populations encountered in other healthcare environments.

### Observational Data

The dataset contains observational historical records. Associations identified by Machine Learning models should not be interpreted as causal relationships.

### Clinical Validity

The resulting models will not be clinically validated and must not be considered suitable for clinical deployment.

---

## 13. Reproducibility

The project will document the complete data and Machine Learning workflow to facilitate reproducibility.

The documentation will include:

- Dataset source.
- Dataset acquisition instructions.
- Dataset version or retrieval date when applicable.
- Python version.
- Project dependencies.
- Data preprocessing steps.
- Feature engineering procedures.
- Model configuration.
- Evaluation methodology.
- Explainability methodology.
- Fairness evaluation methodology.

The raw dataset will not be unnecessarily committed to the Git repository.

Instead, the repository will provide instructions for obtaining the dataset from its official source.

All transformations applied to the data will be implemented through reproducible code.

The final project will aim to allow another user to recreate the main analysis and Machine Learning workflow using the documented environment and instructions.