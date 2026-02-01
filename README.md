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
- Correlation heatmaps - Class distribution - Categorical feature analysis
- Sex vs Target - Chest pain types
- Thalassemia - Major vessels visualization

### 2. **Data Preprocessing**
- Raw data → Remove duplicates → One-hot encode categorical variables (cp, thal, ca)
→ Stratified train/test split (80/20) → StandardScaler


### 3. **Model Pipeline**
Logistic Regression:
├── RandomizedSearchCV (solver, penalty, C, l1_ratio)
└── Pipeline(StandardScaler + LogisticRegression)

Random Forest:
├── RandomizedSearchCV (n_estimators, max_depth, min_samples_split/leaf)
└── class_weight='balanced'


### 4. **Evaluation & Optimization**
- 5-fold StratifiedKFold cross-validation
- Multiple metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- Precision-recall curve threshold tuning for optimal F1-score

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create environment
conda create -n heart-disease python=3.9
conda activate heart-disease

# Install dependencies
pip install -r requirements.txt
```
### Models
| Model               | Preprocessing  | Key Features                           |
| ------------------- | -------------- | -------------------------------------- |
| Logistic Regression | StandardScaler | Interpretability, clinical odds ratios |
| Random Forest       | None           | Robustness, feature importance         |

### Results
Logistic Regression (threshold: 0.630):
Confusion Matrix: [, ][1][2]
Precision: 93%, Recall: 82% for positive cases

Random Forest (threshold: 0.451):
Confusion Matrix: [, ][3][4]
Precision: 88%, Recall: 85% for positive cases

### Visualizations
| Visualization                  | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| medical_feature_importance.png | Top 10 Logistic Regression features (coefficients) |
| feature_importance.png         | Top 10 Random Forest features (Gini importance)    |
| Correlation heatmap            | Feature relationships                              |
| Class distributions            | Target balance analysis                            |
