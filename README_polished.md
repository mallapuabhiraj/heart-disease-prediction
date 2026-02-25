# ❤️ Heart Disease Prediction System

### 🚀 Clinically-Inspired Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![PR-AUC](https://img.shields.io/badge/PR--AUC-0.91-success)

------------------------------------------------------------------------

## 🫀 Overview

A professionally engineered machine learning system designed to predict
the presence of heart disease using structured clinical data from the
UCI Cleveland dataset.

This project is built with:

✅ Leakage-free preprocessing\
✅ Clinically grounded feature engineering\
✅ Robust cross-validation\
✅ Transparent model comparison\
✅ Clean, production-oriented structure

------------------------------------------------------------------------

## 🎯 Problem Statement

Cardiovascular disease is one of the leading causes of global mortality.

This project builds a binary classifier to predict:

-   **0 → No Heart Disease**
-   **1 → Presence of Heart Disease**

The focus is not just accuracy --- but *methodological correctness and
interpretability*.

------------------------------------------------------------------------

## 📂 Dataset

-   📊 \~303 patient records\
-   🏥 Structured clinical attributes\
-   🎯 Binary target classification

Key features include:

-   Age\
-   Chest pain type (cp)\
-   Resting blood pressure (trestbps)\
-   Cholesterol (chol)\
-   Max heart rate achieved (thalach)\
-   ST depression (oldpeak)\
-   Major vessels count (ca)\
-   Thalassemia (thal)\
-   Exercise-induced angina (exang)

------------------------------------------------------------------------

## 🧠 Engineering Approach

### 🔹 Leakage-Free Pipeline

-   Custom preprocessing inside `sklearn.Pipeline`
-   `OneHotEncoder(handle_unknown='ignore')`
-   Stratified cross-validation
-   No transformations outside CV

### 🔹 Clinically Motivated Feature Engineering

``` python
hr_ratio = thalach / (220 - age)
exercise_risk = exang * oldpeak
severity_score = ca + oldpeak
```

Additional engineered signals: - ST depression severity flags\
- Age risk bands\
- Exercise interaction metrics

All features are medically motivated --- not arbitrary polynomial
expansions.

------------------------------------------------------------------------

## 🤖 Models Evaluated

  Model                    Accuracy     Recall       ROC-AUC      PR-AUC
  ------------------------ ------------ ------------ ------------ ------------
  🥇 **SVM**               **0.8689**   **0.9091**   0.8842       0.9084
  🥈 Logistic Regression   0.8525       0.8788       **0.8918**   **0.9112**
  🥉 Random Forest         0.8197       0.8485       0.8820       0.9058

### 🏆 Best Performing Model

**SVM** achieved the highest recall --- critical in medical screening
scenarios where minimizing false negatives is essential.

Logistic Regression demonstrated the strongest probabilistic ranking
performance (highest ROC-AUC & PR-AUC).

------------------------------------------------------------------------

## 📊 Why This Project Stands Out

✨ Proper validation methodology\
✨ Realistic performance (no inflated metrics)\
✨ Clean architecture\
✨ Interpretability-focused modeling\
✨ Reproducible experimentation

This is not a "toy notebook project."\
It reflects production-grade ML discipline.

------------------------------------------------------------------------

## 🏗 Project Structure

    ├── data/
    ├── notebooks/
    ├── src/
    │   ├── preprocess.py
    │   ├── train.py
    │   └── evaluate.py
    ├── models/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   🐍 Python\
-   📦 Scikit-learn\
-   🧮 NumPy\
-   🐼 Pandas\
-   📈 Matplotlib

------------------------------------------------------------------------

## 🚀 Future Improvements

-   Nested cross-validation\
-   Probability calibration curves\
-   Threshold optimization for recall prioritization\
-   SHAP-based interpretability\
-   REST API deployment

------------------------------------------------------------------------

## 🧭 Key Takeaways

-   Correct methodology \> Inflated metrics\
-   Simple models can outperform complex ones on structured medical
    data\
-   Feature engineering should be domain-grounded\
-   Small datasets demand strict validation discipline

------------------------------------------------------------------------

## 📌 Final Note

Built with curiosity, rigor, and a bias toward clean engineering.

Because in healthcare ML --- **precision matters.** 💙
