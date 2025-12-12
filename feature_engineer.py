import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.required_columns = [
            'trans_date_trans_time', 'dob', 'amt',
            'lat', 'long', 'merch_lat', 'merch_long', 'city_pop'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        allowed = set(self.required_columns)
        df = df[[col for col in df.columns if col in allowed]]
        for col in self.required_columns:
            if col not in df.columns:
                df[col] = np.nan

        df['transaction_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

        df['transaction_hour'] = df['transaction_time'].dt.hour.fillna(0)
        df['transaction_day'] = df['transaction_time'].dt.day.fillna(0)
        df['transaction_month'] = df['transaction_time'].dt.month.fillna(0)
        df['age'] = ((df['transaction_time'] - df['dob']).dt.days // 365).fillna(0)
        df['amt_log'] = np.log1p(df['amt']).fillna(0)

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        for col in ['lat', 'long', 'merch_lat', 'merch_long', 'city_pop']:
            df[col] = df[col].fillna(0)

        df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

        feature_cols = [
            'amt', 'amt_log', 'transaction_hour', 'transaction_day', 'transaction_month',
            'age', 'distance', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop'
        ]
        return df[feature_cols]
