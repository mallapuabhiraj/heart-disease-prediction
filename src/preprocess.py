import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class Preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_transformed = self._transform(X)
        self.feature_names_ = X_transformed.columns.tolist()  # remembers train columns
        return self

    def transform(self, X):
        X_transformed = self._transform(X)
        return X_transformed.reindex(columns=self.feature_names_, fill_value=0)  # aligns test to train

    def _transform(self, X):
        X = X.copy()

        # 1. Heart rate achievement ratio
        X['hr_ratio'] = X['thalach'] / (220 - X['age'] + 1)
        # 2. ST depression severity flags
        X['st_depression_high'] = (X['oldpeak'] > 2).astype(int)
        X['st_depression_mild'] = ((X['oldpeak'] > 1) & (X['oldpeak'] <= 2)).astype(int)
        # 3. Exercise combined risk
        X['exercise_risk'] = X['exang'] * X['oldpeak']
        # 4. Vessel + ST depression severity
        X['severity_score'] = X['ca'] + X['oldpeak']
        # 5. Age risk band (ordinal encoded)
        X['age_risk_band'] = pd.cut(
            X['age'], bins=[0, 45, 55, 65, 120], labels=[0, 1, 2, 3]
        ).astype(int)
        X['age_thalach'] = X['age'] * X['thalach']
        X['chol_age'] = X['chol'] / (X['age'] + 1)
        X['age_chol_ratio'] = X['age'] / (X['chol'] + 1)
        X['oldpeak_slope'] = X['oldpeak'] * X['slope']
        X['high_risk_profile'] = (
            ((X['cp'] == 1) | (X['cp'] == 2)) &
            ((X['thal'] == 1) | (X['thal'] == 2))
        ).astype(int)

        return pd.get_dummies(X, drop_first=True)
