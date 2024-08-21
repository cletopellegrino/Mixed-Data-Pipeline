from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class DynamicColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder()

    def fit(self, X, y=None):
        self.categorical_features = [col for col in X.columns if "cat__" in col]
        self.encoder.fit(X[self.categorical_features])
        return self

    def transform(self, X):
        X_encoded = self.encoder.transform(X[self.categorical_features])
        X_remaining = X.drop(columns=self.categorical_features, errors='ignore')
        return np.hstack([X_remaining, X_encoded.toarray()])