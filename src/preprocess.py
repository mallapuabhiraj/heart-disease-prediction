
def preprocess(heart):
    heart = heart.copy()

    # 1. Heart rate achievement ratio
    heart['hr_ratio'] = heart['thalach'] / (220 - heart['age'] + 1)

    # 2. ST depression severity flags
    heart['st_depression_high'] = (heart['oldpeak'] > 2).astype(int)
    heart['st_depression_mild'] = (
        (heart['oldpeak'] > 1) & (heart['oldpeak'] <= 2)
    ).astype(int)

    # 3. Exercise combined risk
    heart['exercise_risk'] = heart['exang'] * heart['oldpeak']

    # 4. Vessel + ST depression severity
    heart['severity_score'] = heart['ca'] + heart['oldpeak']

    # 5. Age risk band (ordinal encoded)
    heart['age_risk_band'] = pd.cut(
        heart['age'],
        bins=[0, 45, 55, 65, 120],
        labels=[0, 1, 2, 3]
    ).astype(int)

    heart['age_thalach'] = heart['age'] * heart['thalach']
    heart['chol_age'] = heart['chol'] / (heart['age'] + 1)
    heart['age_chol_ratio'] = heart['age'] / (heart['chol'] + 1)
    heart['oldpeak_slope'] = heart['oldpeak'] * heart['slope']

    heart['high_risk_profile'] = (
        ((heart['cp'] == 1) | (heart['cp'] == 2)) &
        ((heart['thal'] == 1) | (heart['thal'] == 2))
    ).astype(int)
    return heart
class Preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_transformed = self._transform(X)
        self.feature_names_ = X_transformed.columns.tolist()  # added - remembers train columns
        return self

    def transform(self, X):
        X_transformed = self._transform(X)
        return X_transformed.reindex(columns=self.feature_names_, fill_value=0)  # added - aligns test to train

    def _transform(self, X):         # added - extracted helper
        X = preprocess(X.copy())
        return pd.get_dummies(X, drop_first=True)   # added - encoding now lives here
