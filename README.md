# 🫀 Heart Disease Prediction

> *Because a missed diagnosis isn't a statistic — it's a person.*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ja3Nq4JUDv1eladVbhZSnpZe6oavZO7B)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 🩺 The Problem

Cardiovascular disease is the **#1 cause of death globally**, claiming **17.9 million lives** annually. Early detection is the single most effective intervention — yet clinical diagnosis relies heavily on physician intuition and remains stubbornly prone to false negatives.

**The cost of a missed positive is a human life.**

This project builds an end-to-end machine learning pipeline that predicts heart disease presence from 13 routine clinical features, optimizing specifically for **recall** — because in medicine, a false negative is catastrophically worse than a false alarm.

---

## 💡 How We Solved It

### 1. Data
- Cleveland Heart Disease dataset — **303 patients**, 13 clinical features, binary target
- After deduplication: **302 patients**, **54.5% positive class** (naturally balanced — no SMOTE needed)
- 80/20 stratified train/test split → **241 training**, **61 test** samples

### 2. Feature Engineering
Rather than trusting raw features blindly, we engineered **10 clinically motivated features** on top of the original 13, encapsulated inside a reusable sklearn `TransformerMixin`:

| Engineered Feature | Formula | Clinical Rationale |
|---|---|---|
| `severity_score` | `ca + oldpeak` | Combines vessel blockage + ST depression into a single risk index |
| `high_risk_profile` | `(cp∈{1,2}) AND (thal∈{1,2})` | Flags atypical chest pain + abnormal thalassemia together |
| `hr_ratio` | `thalach / (220 − age)` | Normalizes max HR against the age-predicted maximum |
| `exercise_risk` | `exang × oldpeak` | Amplifies risk when angina meets ST depression under exertion |
| `st_depression_high` | `oldpeak > 2` | Hard clinical threshold for severe ST depression |
| `age_risk_band` | Bins: 0–45, 45–55, 55–65, 65+ | Ordinal age-based cardiovascular risk stratification |

> 💡 **2 of the top 3 predictors** in the final model were engineered features — validating this effort completely.

### 3. Modeling Pipeline
Three models tuned end-to-end inside sklearn **Pipelines** (`Preprocess → StandardScaler → Model`), ensuring zero data leakage:

| Model | Search Strategy | Candidates | CV F1 | CV Std | Verdict |
|---|---|---|---|---|---|
| SVM (RBF) | GridSearchCV | 32 | 0.8640 | ±0.063 | Best mean |
| Random Forest | RandomizedSearchCV | 50 | 0.8633 | ±0.032 | Most stable |
| Logistic Regression | RandomizedSearchCV | 100 | 0.8577 | ±0.074 | Most volatile |

**Best hyperparameters found:**
- SVM → `C=1, kernel=rbf, gamma=0.01`
- RF → `n_estimators=250, max_depth=4, max_features=log2, min_samples_leaf=4`
- LR → `solver=saga, penalty=elasticnet, l1_ratio=0.2, C=0.0215`

### 4. Threshold Tuning
The default 0.5 threshold optimizes accuracy, not recall. We swept the precision-recall curve for each model to find the **highest threshold that still achieves ≥ 90% recall**, then picked the best F1 within that constraint.

---

## 📊 Results

Final evaluation on the held-out test set (**61 patients, stratified**):

| Model | Accuracy | Recall | False Negatives | False Positives | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **SVM** ✅ | **0.869** | **0.909** | **3** | **5** | **0.882** | 0.884 | 0.908 |
| Logistic Regression | 0.820 | 0.909 | 3 | 8 | 0.845 | 0.892 | 0.911 |
| Random Forest | 0.803 | 0.909 | 3 | 9 | 0.833 | 0.882 | 0.906 |

All three models hit identical recall after tuning. **SVM wins the tiebreaker decisively** — fewest false positives (5 vs 8 vs 9), highest accuracy, highest F1. Fewer false positives means fewer healthy patients wrongly alarmed, which matters both clinically and ethically.

---

## 🔍 What the Model Learned

Both LR and RF feature importance independently ranked the same top predictors — strong evidence these signals are real:

```
Rank  Feature              Source        Both Models Agree?
────────────────────────────────────────────────────────
 1    cp                   Original      ✅ #1 in both
 2    severity_score       Engineered    ✅ #2 in both
 3    high_risk_profile    Engineered    ✅ Top 5 in both
 4    exang                Original      ✅ Top 5 in both
 5    ca                   Original      ✅ Top 5 in both
```

---

## 💻 Quick Start

```bash
git clone https://github.com/mallapuabhiraj/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

**Run a prediction:**
```python
import joblib
import pandas as pd

model = joblib.load('Heart_Disease_SVM.joblib')

sample = pd.DataFrame([{
    'age': 54, 'sex': 1, 'cp': 2, 'trestbps': 130,
    'chol': 250, 'fbs': 0, 'restecg': 1, 'thalach': 155,
    'exang': 0, 'oldpeak': 1.5, 'slope': 2, 'ca': 0, 'thal': 2
}])

pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

print(f"Heart Disease: {'Yes ⚠️' if pred else 'No ✅'}")
print(f"Probability:   {prob:.2%}")
```

**Requirements:**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

---

## 🗂️ Project Structure

```
heart-disease-prediction/
│
├── heart.csv                           # Cleveland dataset
├── Heart_Disease_Prediction.ipynb      # Main notebook
├── requirements.txt                    # Dependencies
│
├── models/
│   ├── Heart_Disease_SVM.joblib        # ✅ Final model
│   ├── Heart_Disease_RandomForest.joblib
│   └── Heart_Disease_Logistic.joblib
│
└── plots/
    ├── roc_curve.png
    ├── feature_importance.png          # Random Forest
    └── medical_feature_importance.png  # Logistic Regression
```

---

## 🧠 Key Takeaways

- **Threshold tuning > model selection** when recall is the constraint — all three models converged to identical recall, making false positive rate the real differentiator
- **Feature engineering paid off** — 2 of the top 3 predictors were hand-crafted, not original features
- **Stability matters** — RF had the lowest CV std (±0.032) but SVM generalized better on the test set; small datasets can flip rankings
- **Pipelines are non-negotiable** — encapsulating preprocessing inside the pipeline was the difference between clean cross-validation and silent data leakage

---

## ⚠️ Limitations

- **Small dataset** — 302 patients is enough to build a proof of concept, but insufficient for clinical use. Reported metrics (e.g. `0.869` accuracy) carry wide confidence intervals on just 61 test samples
- **Single cohort** — Cleveland dataset only; performance on other populations, ethnicities, or clinical settings is unknown
- **Not a diagnostic tool** — this is an educational ML project. Do not use model outputs as a substitute for clinical judgment

---

## 📁 Dataset

[UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) — Cleveland subset, 303 instances, 14 attributes. Courtesy of Robert Detrano, V.A. Medical Center, Long Beach.

---

<div align="center">
  <sub>Built with 🖤 and too much coffee by <a href="https://github.com/mallapuabhiraj">mallapuabhiraj</a></sub>
</div>
