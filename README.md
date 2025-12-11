# Fraud Detection App

This repository contains a **Fraud Detection pipeline** built using **CatBoost**. It includes automatic feature engineering and preprocessing, allowing predictions on any dataset with the same structure. You can run the model in **Jupyter Notebook** or through a **Streamlit web app**.

---

## Features

- Automatic feature engineering:
  - Extracts transaction hour, day, month
  - Calculates age from date of birth
  - Computes Haversine distance between user and merchant locations
  - Log transformation of transaction amount
- Scaled numeric features using StandardScaler
- Predicts fraud probabilities and binary labels
- Maintains the same threshold for fraud detection: **0.7**
- Streamlit web app for easy CSV upload and prediction download

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Eyiwunmi/Fraud_Detection_App.git
cd Fraud_Detection_App

## Project Structure
Fraud_Detection_App/
├── streamlit_app.py          # Streamlit application
├── fraud_detection_pipeline.pkl  # Saved pipeline + threshold
├── df_train.csv              # Raw training data (optional)
├── df_test.csv               # Raw test data (optional)
├── requirements.txt          # Python dependencies
└── README.md


## Installation
1. Clone the repository:
```bash
git clone https://github.com/Eyiwunmi/Fraud_Detection_App.git
cd Fraud_Detection_App

2. Activate virtual environment
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Windows CMD
.\venv\Scripts\activate.bat
# macOS/Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Streamlit Web App

Run the Streamlit app: streamlit run streamlit_app.py
Open this URL to use the app: http://localhost:8501/

Upload any CSV file with your transaction data
The app will automatically preprocess, engineer features, and make predictions
Download the predictions as a CSV

# Link to the Dataset: https://drive.google.com/drive/folders/1a9Ijv5E3GdWj2ZgVaeAGq_Zyz7vHb6OC?usp=sharing

# Challenges

Managing large files (datasets and venv) for GitHub.

Dependency management and environment setup.

Saving/loading machine learning pipeline properly with joblib.

# Notes

Ensure your CSV columns match the required columns for feature engineering:

trans_date_trans_time, dob, amt, lat, long, merch_lat, merch_long, city_pop

Missing columns will automatically be filled with NaN or 0 to prevent errors

Threshold for fraud classification is 0.7, adjustable in the pipeline if needed

# Future Work

Add Git LFS for large datasets.

Improve model accuracy with more features.

Deploy on Streamlit Community Cloud.

Add logging, error handling, and tests.

Author
Sarah Eyiwunmi Olarinde
