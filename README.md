# physionet
ML-based Sepsis Prediction using Random Forest &amp; Streamlit
# 🩺 Sepsis Prediction Using Machine Learning

A machine learning project for predicting sepsis risk from clinical data using a Random Forest Classifier.

> ⚠️ This project is for educational purposes only and is not a medical diagnostic tool.

---

## 📌 Project Overview

The project uses the `physionet_sepsis` dataset to build a machine learning model for sepsis risk prediction.

The workflow includes:
- Data preprocessing and missing value handling
- Patient-based train/test splitting
- Training and comparing classification models
- Feature importance analysis
- Building an interactive Streamlit application

---

## 📊 Dataset

- **Dataset:** `physionet_sepsis`
- **Rows:** 1,552,210
- **Original features:** 44
- **Target:** `SepsisLabel`

---

## 🤖 Models & Results

The following models were compared:

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting

**Random Forest** was selected as the final model based on the highest F1 Score.

| Metric | Score |
|---|---:|
| Accuracy | 85.92% |
| Precision | 6.82% |
| Recall | 50.82% |
| F1 Score | 12.03% |
| ROC-AUC | 76.02% |

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit interface for entering clinical data and viewing the prediction probability.

The application includes:
- Clinical Data
- Analysis
- Prediction History
- About

The model uses **26 features**, while the interface displays the **14 most important features** for a simpler user experience.

---

## 🛠️ Technologies

Python • Pandas • NumPy • Scikit-learn • Joblib • Streamlit • Google Colab

---

## 🚀 How to Run

```bash
git clone YOUR_REPOSITORY_URL
cd physionet-sepsis
pip install streamlit pandas numpy scikit-learn joblib
python -m streamlit run app.py


⚠️ ## Disclaimer
This project is for educational purposes only. The prediction is a machine learning estimate and not a medical diagnosis.
