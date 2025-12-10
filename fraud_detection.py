# feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Feature Engineering Transformer for Fraud Detection.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Log transformation for amount
        if "amt" in X.columns:
            X["amt_log"] = np.log1p(X["amt"])

        # Timestamp features
        if "trans_date_trans_time" in X.columns:
            dt = pd.to_datetime(X["trans_date_trans_time"], errors='coerce')
            X["transaction_hour"] = dt.dt.hour
            X["transaction_day"] = dt.dt.day
            X["transaction_month"] = dt.dt.month

        # Distance between customer and merchant
        if all(col in X.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
            X["distance"] = np.sqrt(
                (X["lat"] - X["merch_lat"])**2 + (X["long"] - X["merch_long"])**2
            )

        return X
