# physionet
ML-based Sepsis Prediction using Random Forest &amp; Streamlit

🩺 Sepsis Prediction Using Machine Learning

1. Project Overview

Sepsis is a serious medical condition that can develop when the body's response to an infection causes tissue and organ damage. Early identification of sepsis can support timely medical intervention.

This project applies Machine Learning techniques to clinical data to build a binary classification model for predicting sepsis. The project covers the complete process from data exploration and preprocessing to model training, evaluation, and integration into a Streamlit application.

«Note: This project is developed for educational purposes and is not intended for medical diagnosis, treatment, or clinical decision-making.»

---

2. Project Objective

The main objective is to develop a Machine Learning model that predicts whether a patient is likely to be classified with sepsis based on available clinical measurements.

The project focuses on:

- Preparing and analyzing a large clinical dataset.
- Handling missing and invalid data.
- Training and comparing different classification models.
- Evaluating models using metrics suitable for imbalanced classification.
- Identifying important features used by the final model.
- Integrating the trained model into an interactive Streamlit application.

---

3. Dataset

The project uses the PhysioNet Sepsis dataset available through OpenML.

Dataset ID: 46817
Number of records: 1,552,210
Number of original features: 44
Target variable: "SepsisLabel"

"SepsisLabel" is a binary target indicating whether the corresponding clinical record is associated with sepsis.

The dataset contains patient information, vital signs, laboratory measurements, ICU information, and time-related data.

Main Feature Groups

Group| Examples
Vital Signs| Heart Rate, Temperature, Blood Pressure, Respiration Rate, Pulse Oximetry
Laboratory Measurements| pH, PaCO2, Glucose, Creatinine, Hemoglobin, Platelet Count
Patient Information| Age, Gender
ICU Information| MICU, SICU
Time| Hour

The dataset contains a considerable amount of missing data, especially among laboratory measurements, which makes preprocessing an important part of the project.

---

4. Data Exploration & Preprocessing

Before training the models, the dataset was inspected to understand its structure, data quality, missing values, duplicates, and target distribution.

The preprocessing steps were performed in a way that avoids using information from the test set during training.

4.1 Data Inspection

The dataset was examined using:

- "shape"
- "info()"
- "describe()"
- Missing-value analysis
- Duplicate-value analysis
- Target distribution
- Invalid-value inspection

This helped identify the main data-quality issues before model training.

4.2 Removing Unnecessary Features

"Patient_ID" was removed because it is an identifier rather than a predictive clinical feature.

"ICU_length_of_stay" was also removed because it represents information that can introduce leakage when making predictions.

4.3 Patient-Based Train-Test Split

The data was divided into training and testing sets using patient-level separation.

This ensures that records from the same patient are not present in both sets, reducing the risk of data leakage.

The split used:

- Training set: 80%
- Testing set: 20%
- "random_state = 42"

4.4 Handling Highly Missing Features

Features with more than 95% missing values were removed based on the training data.

This reduced the number of features while avoiding unnecessary imputation for variables with very limited information.

4.5 Missing Value Imputation

The remaining missing values were handled using:

SimpleImputer(strategy="median")

The imputer was fitted on the training data and then applied to both the training and testing sets.

4.6 Feature Scaling

"StandardScaler" was used for models that require feature scaling, such as Logistic Regression.

The scaler was fitted only on the training data and then applied to the test data.

After preprocessing, 26 features were used for model training.

---

5. Machine Learning Models

After preprocessing, four classification models were trained and evaluated.

5.1 Logistic Regression

Logistic Regression was used as a baseline classification model.

Because the target classes are highly imbalanced, "class_weight="balanced"" was used to give more importance to the minority class.

LogisticRegression(class_weight="balanced",random_state=42)

5.2 Decision Tree

Decision Tree was included to provide a non-linear model that can capture relationships between features without requiring linear assumptions.

Class balancing was also applied:

DecisionTreeClassifier(class_weight="balanced",random_state=42)

5.3 Random Forest

Random Forest was used as an ensemble model consisting of multiple decision trees.

The final configuration was:

RandomForestClassifier(n_estimators=90,max_depth=10,class_weight="balanced",random_state=42)

The model was selected as the final model after comparing the performance of all tested models.

5.4 HistGradientBoosting

HistGradientBoosting was included because it can efficiently handle large datasets and learn non-linear patterns.

The configuration used was:

HistGradientBoostingClassifier(max_iter=100,learning_rate=0.1,max_leaf_nodes=31,random_state=42)

---

6. Model Evaluation & Results

Since the dataset is highly imbalanced, accuracy alone is not sufficient to evaluate the models.

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Results

Model| Accuracy| Precision| Recall| F1 Score| ROC-AUC
Logistic Regression| 75.49%| 4.44%| 58.12%| 8.24%| 71.45%
Decision Tree| 79.53%| 4.99%| 54.38%| 9.14%| 70.21%
Random Forest| 85.92%| 6.82%| 50.82%| 12.03%| 76.02%
HistGradientBoosting| 98.09%| 23.89%| 0.46%| 0.90%| 76.84%

The Random Forest achieved the highest F1 Score among the tested models and was therefore selected as the final model.

The results also show the effect of class imbalance. Although HistGradientBoosting achieved high accuracy, its recall for the positive class was very low, resulting in a very low F1 Score.

The Random Forest model provided a more useful balance between Precision and Recall among the tested models.

---

7. Feature Importance Analysis

After selecting Random Forest, feature importance was analyzed to understand which features contributed most to the model's predictions.

The most important features included:

Feature| Importance
Hour| 0.5844
Heart Rate| 0.0856
Respiration Rate| 0.0596
Temperature| 0.0516
Mean Arterial Pressure| 0.0307
FiO2| 0.0242
Systolic Blood Pressure| 0.0196
Age| 0.0195
Diastolic Blood Pressure| 0.0169
pH| 0.0157
PaCO2| 0.0112
Excess Bicarbonate| 0.0093
Pulse Oximetry| 0.0091
MICU| 0.0087

"Hour" had a substantially higher importance than the other features in this model.

Feature importance describes how the trained model uses the available variables; it does not establish that a feature directly causes sepsis.

---

8. Streamlit Application

The trained Random Forest model was integrated into a Streamlit application to provide a simple interactive interface.

The application is organized into several sections:

Home

Provides an overview of the project and its purpose.

Clinical Data

Allows the user to enter clinical measurements used for prediction.

Prediction Result

Displays the model's prediction based on the provided inputs.

Analysis

Provides information related to the prediction and important model features.

History

Allows previous prediction results to be viewed during the application session.

About

Provides information about the project, model, and its educational purpose.

User Interface Features

For simplicity, the application displays 14 important clinical features to the user instead of requiring all 26 model features to be entered manually.

The trained model itself uses 26 features. The remaining features are handled by the application's preprocessing pipeline before the prediction is made.

---

9. Model Deployment

The trained model and preprocessing components were saved using "Joblib" and loaded by the Streamlit application.

The main files used by the application are:

rf_model.pkl
imputer.pkl
feature_names.pkl

The prediction process follows this sequence:

User Input
    ↓
Feature Preparation
    ↓
Saved Imputer
    ↓
Prepared Feature Vector
    ↓
Saved Random Forest Model
    ↓
Prediction
    ↓
Streamlit Result

This allows the application to use the same trained model and preprocessing configuration without retraining the model each time the application runs.

---

10. Project Structure

physionet-sepsis/
│
├── app.py
├── rf_model.pkl
├── imputer.pkl
├── feature_names.pkl
├── doctor_patient.png
└── README.md

- "app.py" — Streamlit application.
- "rf_model.pkl" — trained Random Forest model.
- "imputer.pkl" — fitted preprocessing imputer.
- "feature_names.pkl" — feature names used by the model.
- "doctor_patient.png" — application image.
- "README.md" — project documentation.

---

11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- Google Colab

---

12. Limitations

The project has several limitations that should be considered when interpreting the results:

- The dataset contains substantial missing values.
- The target variable is highly imbalanced.
- The positive-class Precision and F1 Score remain relatively low.
- The model results are specific to the dataset and preprocessing approach used.
- Feature importance should not be interpreted as clinical causation.
- The Streamlit application is intended for educational demonstration rather than real-world clinical use.

---

13. Conclusion

This project followed a complete Machine Learning workflow for sepsis prediction, starting with clinical data exploration and preprocessing, followed by training and evaluation of multiple classification models.

Four models were tested, and Random Forest was selected as the final model based on its F1 Score among the tested approaches. Feature importance was then analyzed to understand how the trained model used the available variables.

Finally, the trained model was integrated into a Streamlit application, providing an interactive interface for entering clinical data and viewing predictions.

The project demonstrates the practical application of Machine Learning to a large clinical dataset while also highlighting important challenges such as missing data, class imbalance, and the limitations of model performance in a healthcare context.
