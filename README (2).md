# Heart Disease Prediction System

## Clinically-Inspired Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![PR-AUC](https://img.shields.io/badge/PR--AUC-0.91-success)

------------------------------------------------------------------------

## Overview

A rigorously engineered machine learning system for predicting the
presence of heart disease using structured clinical data from the UCI
Cleveland dataset.

Designed with production discipline in mind:

-   Leakage-free preprocessing\
-   Clinically grounded feature engineering\
-   Stratified cross-validation\
-   Transparent model benchmarking\
-   Reproducible experimentation

Clean methodology. Realistic metrics. No shortcuts.

------------------------------------------------------------------------

## Problem Statement

Cardiovascular disease remains a leading global health risk.

This project builds a binary classifier to predict:

-   **0 -\> No Heart Disease**\
-   **1 -\> Presence of Heart Disease**

The emphasis is on correctness, interpretability, and evaluation
integrity --- not leaderboard inflation.

------------------------------------------------------------------------

## Dataset

-   \~303 patient records\
-   Structured clinical attributes\
-   Binary classification target

Core features include age, chest pain type, resting blood pressure,
cholesterol, maximum heart rate, ST depression, vessel count,
thalassemia status, and exercise-induced angina.

------------------------------------------------------------------------

## Engineering Approach

### Leakage-Free Pipeline

All preprocessing is encapsulated within `sklearn.Pipeline`, ensuring
transformations occur strictly inside cross-validation folds.

-   `OneHotEncoder(handle_unknown='ignore')`\
-   Stratified K-fold validation\
-   No data leakage

### Clinically Motivated Feature Engineering

``` python
hr_ratio = thalach / (220 - age)
exercise_risk = exang * oldpeak
severity_score = ca + oldpeak
```

Additional signals:

-   ST depression severity indicators\
-   Age-based risk segmentation\
-   Exercise interaction metrics

Every engineered feature is medically motivated --- no arbitrary feature
explosion.

------------------------------------------------------------------------

## Model Benchmarking

  Model                 Accuracy   Recall   ROC-AUC   PR-AUC
  --------------------- ---------- -------- --------- --------
  **SVM**               0.8689     0.9091   0.8842    0.9084
  Logistic Regression   0.8525     0.8788   0.8918    0.9112
  Random Forest         0.8197     0.8485   0.8820    0.9058

### Model Selection Insight

-   **SVM** achieved the highest recall --- critical in screening
    scenarios where false negatives are costly.\
-   Logistic Regression demonstrated the strongest ranking performance
    (ROC-AUC and PR-AUC).

Simple, well-validated models outperform complexity when the data
demands discipline.

------------------------------------------------------------------------

## Project Structure

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

## Technology Stack

-   Python\
-   Scikit-learn\
-   NumPy\
-   Pandas\
-   Matplotlib

------------------------------------------------------------------------

## Roadmap

-   Nested cross-validation\
-   Probability calibration\
-   Decision threshold optimization\
-   SHAP-based interpretability\
-   REST API deployment

------------------------------------------------------------------------

## Final Note

Healthcare machine learning demands rigor over flash.

This project reflects disciplined validation, domain-aware feature
design, and pragmatic modeling choices.

Precision matters.
