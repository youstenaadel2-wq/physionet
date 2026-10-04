# physionet
ML-based Sepsis Prediction using Random Forest &amp; Streamlit
 # 🩺 Sepsis Prediction Using Machine Learning

## 1. Project Overview

Sepsis is a serious medical condition that can develop when the body's response to an infection causes tissue and organ damage. Early identification of sepsis can support timely medical intervention.

This project applies Machine Learning techniques to clinical data to build a binary classification model for sepsis prediction. The workflow covers data exploration, data cleaning, patient-level splitting, missing-value handling, correlation analysis, outlier treatment, class-imbalance handling, model training, validation, threshold tuning, final evaluation, and model deployment using Streamlit.

> **Note:** This project is developed for educational purposes only and is not intended for medical diagnosis, treatment, or clinical decision-making.

---

## 2. Project Objective

The main objective is to develop a Machine Learning model that can identify records associated with sepsis from clinical measurements.

The project focuses on:

- Exploring and cleaning a large clinical dataset.
- Preventing patient-level data leakage.
- Handling missing and invalid values.
- Removing features with extremely high missing rates.
- Reducing highly correlated features.
- Handling extreme values using outlier clipping.
- Addressing severe class imbalance in the training data.
- Comparing different classification models.
- Selecting classification thresholds using validation data.
- Evaluating the models using metrics suitable for imbalanced classification.
- Analyzing influential features.
- Saving the trained model and preprocessing components for deployment.

---

## 3. Dataset

The project uses the PhysioNet Sepsis dataset available through OpenML.

### Dataset Information

| Property | Value |
|---|---:|
| OpenML Dataset ID | 46817 |
| Records | 1,552,210 |
| Original Columns | 44 |
| Target | `SepsisLabel` |
| Negative Class | 98.20% |
| Positive Class | 1.80% |

The target is binary:

| Value | Meaning |
|---|---|
| `0` | No Sepsis |
| `1` | Sepsis |

The dataset contains vital signs, laboratory measurements, patient information, ICU-related variables, and other clinical measurements.

Examples include:

- Heart Rate
- Pulse Oximetry
- Temperature
- Blood Pressure
- Respiration Rate
- pH
- Glucose
- Creatinine
- Hemoglobin
- Platelet Count
- Age
- Gender
- ICU-related information

A major challenge is the large amount of missing data, particularly in laboratory measurements.

---

## 4. Data Exploration

The dataset was first inspected to understand its structure and data quality.

The following checks were performed:

- Dataset shape
- Column names
- Data types
- Descriptive statistics
- Missing values
- Duplicate records
- Target distribution
- Invalid values

The initial dataset contained **1,552,210 rows** and **44 columns**.

No complete duplicate rows were found.

The target distribution confirmed severe class imbalance, with only about **1.8% positive records**.

---

## 5. Data Cleaning and Preprocessing

The preprocessing workflow was designed to avoid using information from the validation or test sets when fitting preprocessing steps.

### 5.1 Removing Identifier and Redundant Columns

The following columns were removed:

| Column | Reason |
|---|---|
| `Patient_ID` | Identifier, not a clinical predictor |
| `Unnamed: 0` | Redundant index-like column |
| `Hour` | Removed from the current modeling pipeline to avoid relying on the time index |

`ICU_length_of_stay` was also excluded from the modeling features because it can introduce information leakage when making predictions.

### 5.2 Handling Invalid Values

Clearly invalid values were converted to missing values before imputation.

Measurements such as the following were checked for invalid low values:

- Heart Rate
- Blood Pressure
- Temperature
- Respiration Rate
- Age

Binary variables were also checked to ensure that their values were restricted to `0` and `1`.

---

## 6. Patient-Level Train, Validation, and Test Split

Because the dataset contains multiple records for the same patient, a normal row-based split could cause data leakage.

Instead, patients were divided first and their records were assigned to the corresponding datasets.

### Split Strategy

| Stage | Proportion |
|---|---:|
| Train + Validation | 80% |
| Test | 20% |
| Training from Train + Validation | 80% |
| Validation from Train + Validation | 20% |

### Final Dataset Sizes

| Dataset | Records | Positive Records |
|---|---:|---:|
| Training | 993,338 | 17,605 |
| Validation | 248,879 | 4,439 |
| Test | 309,993 | 5,872 |

Patient IDs were checked to ensure there was no overlap between the three sets.

### Patient ID Overlap Check

| Comparison | Overlap |
|---|---:|
| Train ↔ Validation | 0 |
| Train ↔ Test | 0 |
| Validation ↔ Test | 0 |

This reduces the risk of patient-level data leakage.

---

## 7. Handling Missing Values

Missing-value analysis was performed using the training set.

Features with more than **95% missing values** were removed because they contained too little observed information to be reliably imputed.

### Features Removed

| Feature |
|---|
| Direct Bilirubin |
| Fibrinogen |
| Troponin I |
| Total Bilirubin |
| Alkaline Phosphatase |
| AST |
| Lactic Acid |
| Partial Thromboplastin Time |
| Arterial Oxygen Saturation |
| End-tidal CO₂ |
| Phosphate |
| Bicarbonate |
| Chloride |

This removed **13 features**, leaving **27 features** before correlation analysis.

The remaining missing values were handled using:

```python
SimpleImputer(
    strategy="median",
    add_indicator=True
)
