# Heart Disease Prediction ML Pipeline

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ja3Nq4JUDv1eladVbhZSnpZe6oavZO7B)

Comprehensive machine learning pipeline for predicting heart disease using the UCI Heart Disease dataset (303 patients, 14 clinical features). Implements Logistic Regression and Random Forest with hyperparameter tuning, threshold optimization, and feature importance analysis.

## 📊 Table of Contents
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Models](#-models)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Saved Models](#-saved-models)
- [Contributing](#-contributing)
- [License](#-license)

## 🩺 Dataset
**UCI Heart Disease Dataset** - Classic medical benchmark containing:
- **303 samples**, **14 features** including age, sex, chest pain type, cholesterol, resting blood pressure
- Binary target: presence (1) or absence (0) of heart disease
- Real-world clinical data from Cleveland Clinic

## ✨ Features
- Complete ML lifecycle: EDA → Preprocessing → Tuning → Evaluation → Deployment
- Hyperparameter optimization with RandomizedSearchCV
- Cross-validation across multiple metrics
- Precision-recall threshold optimization
- Feature importance analysis (coefficients + Gini importance)
- Production-ready Joblib model artifacts
- Professional visualizations and EDA

## 🛠️ Methodology

### 1. **Exploratory Data Analysis**
