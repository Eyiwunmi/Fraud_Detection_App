# streamlit_app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Fraud Detection", layout="wide")

# ---------------------------
# 1) LOCAL FeatureEngineer used by the app (for uploaded CSVs)
# ---------------------------
from sklearn.base import BaseEstimator, TransformerMixin

class AppFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Lightweight feature engineering used by Streamlit before passing data
    to the model. Must produce the same engineered columns that the model expects.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # ensure raw columns exist
        raw_cols = ['trans_date_trans_time', 'dob', 'amt', 
                    'lat', 'long', 'merch_lat', 'merch_long', 'city_pop']
        for c in raw_cols:
            if c not in df.columns:
                df[c] = np.nan

        # datetimes
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

        # numeric casts with safe fill
        num_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop']
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # basic features
        df['transaction_hour'] = df['trans_date_trans_time'].dt.hour.fillna(0).astype(int)
        df['transaction_day'] = df['trans_date_trans_time'].dt.day.fillna(0).astype(int)
        df['transaction_month'] = df['trans_date_trans_time'].dt.month.fillna(0).astype(int)

        # age
        df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days // 365).fillna(0).astype(int)

        # log amount
        df['amt_log'] = np.log1p(df['amt']).fillna(0)

        # haversine distance (km)
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return R * c

        df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long']).fillna(0)

        # Choose the feature columns expected by your model.
        # IMPORTANT: make sure this list matches what the model expects.
        feature_cols = [
            'amt', 'amt_log', 'transaction_hour', 'transaction_day', 'transaction_month',
            'age', 'distance', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop'
        ]

        # If some feature columns are missing, add them with zeros
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0

        return df[feature_cols]


# ---------------------------
# 2) Utilities: cleaning uploaded CSV
# ---------------------------
REQUIRED_RAW_COLS = [
    'trans_date_trans_time', 'dob', 'amt',
    'lat', 'long', 'merch_lat', 'merch_long', 'city_pop'
]

def clean_uploaded_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning to make sure required raw columns are present & typed."""
    df = df.copy()
    for c in REQUIRED_RAW_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # convert types
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    numeric_cols = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


# ---------------------------
# 3) Load pipeline and threshold
# ---------------------------
@st.cache_resource(ttl=3600)
def load_pipeline(path="fraud_detection_pipeline.pkl"):
    loaded = joblib.load(path)
    # support saved as either pipeline or (pipeline, THRESHOLD)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        pipeline, threshold = loaded
    else:
        pipeline = loaded
        threshold = 0.5
    return pipeline, threshold

try:
    pipeline, THRESHOLD = load_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

# ---------------------------
# 4) Detect pipeline internals to avoid double FE
# ---------------------------
def pipeline_has_internal_feature_engineer(pipeline: Pipeline) -> bool:
    try:
        names = list(pipeline.named_steps.keys())
        # Check by name or estimator class name heuristics
        for name, est in pipeline.named_steps.items():
            lower_name = name.lower()
            cls_name = est.__class__.__name__.lower()
            if 'feature' in lower_name or 'feature' in cls_name or 'engineer' in cls_name:
                return True
    except Exception:
        pass
    return False

def find_scaler_and_model(pipeline: Pipeline):
    """Return (scaler_estimator or None, model_estimator). Heuristics: scaler named 'scaler' or StandardScaler."""
    scaler = None
    model = None
    try:
        for name, est in pipeline.named_steps.items():
            clsname = est.__class__.__name__.lower()
            if 'scaler' in name.lower() or 'standardscaler' in clsname:
                scaler = est
        # model is usually last step
        last_name = list(pipeline.named_steps.keys())[-1]
        model = pipeline.named_steps[last_name]
    except Exception:
        model = pipeline  # fallback: pipeline itself (call .predict_proba)
    return scaler, model


# ---------------------------
# 5) Streamlit UI
# ---------------------------
st.title("Fraud Detection App")
st.markdown("Upload a CSV; app will clean, engineer features, and produce fraud probabilities.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data (first 5 rows):")
    st.dataframe(raw_df.head())

    # Clean raw dataframe (adds missing raw columns and casts types)
    cleaned_raw = clean_uploaded_dataframe(raw_df)

    # Create engineered features in Streamlit
    fe = AppFeatureEngineer()
    engineered_df = fe.transform(cleaned_raw)

    st.write("Preview of engineered features (first 5 rows):")
    st.dataframe(engineered_df.head())

    # Decide how to call the model:
    has_internal_fe = pipeline_has_internal_feature_engineer(pipeline)

    if has_internal_fe:
        # If pipeline includes feature engineering, avoid double-fe by applying model/scaler directly.
        scaler, model = find_scaler_and_model(pipeline)

        # If model is still a pipeline object or doesn't expose predict_proba, fall back:
        if hasattr(model, "predict_proba") and not isinstance(model, Pipeline):
            X_for_model = engineered_df.values if scaler is None else scaler.transform(engineered_df)
            preds_proba = model.predict_proba(X_for_model)[:, 1]
        else:
            # fallback: let pipeline handle it (it should expect raw input) - use cleaned_raw
            preds_proba = pipeline.predict_proba(cleaned_raw)[:, 1]
    else:
        # pipeline expects raw input (it does not contain internal FE) — pass raw cleaned data
        try:
            preds_proba = pipeline.predict_proba(cleaned_raw)[:, 1]
        except Exception:
            # last fallback: pass engineered features (in case pipeline expects engineered features)
            preds_proba = pipeline.predict_proba(engineered_df)[:, 1]

    # Attach results to original raw_df for download and display
    result_df = raw_df.copy()
    result_df["fraud_probability"] = preds_proba
    result_df["fraud_prediction"] = (result_df["fraud_probability"] >= THRESHOLD).astype(int)

    st.subheader("Predictions (preview)")
    st.dataframe(result_df.head())

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions (CSV)", csv_bytes, file_name="predictions.csv")
