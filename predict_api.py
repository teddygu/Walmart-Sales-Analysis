"""
predict_api.py
Simple Flask API that loads the model and serves predictions.
Example POST JSON body:
{
  "Store": "1",
  "Dept": "1",
  "Date": "2012-12-01",
  "IsHoliday": false
}
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

MODEL = joblib.load("models/lgb_model.joblib")
FEATURES = joblib.load("models/feature_list.joblib")
ENCODERS = joblib.load("data/processed/encoders.joblib")

def prepare_single(row):
    # row: dict with Store, Dept, Date, IsHoliday
    df = pd.DataFrame([row])
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['Date'].dt.weekday
    df['day'] = df['Date'].dt.day
    df['us_holiday'] = df['Date'].apply(lambda d: d in __import__('holidays').UnitedStates())
    # encode store/dept using saved categories
    df['Store'] = df['Store'].astype(str)
    df['Dept'] = df['Dept'].astype(str)
    try:
        df['store_code'] = df['Store'].apply(lambda s: ENCODERS['store_categories'].index(s))
    except ValueError:
        # unseen store - use -1
        df['store_code'] = -1
    try:
        df['dept_code'] = df['Dept'].apply(lambda s: ENCODERS['dept_categories'].index(s))
    except ValueError:
        df['dept_code'] = -1
    # For lags/rolling we can't know true recent sales here. We'll set them to median or 0.
    for c in ['lag_1','lag_2','lag_3','lag_52','rolling_4','rolling_8']:
        df[c] = np.nan
    # fallback defaults
    df['days_from_start'] = (df['Date'] - pd.Timestamp("2010-01-01")).dt.days  # arbitrary start
    # Fill NaNs with 0 or median; in real production you'd keep a store+dept recent history
    df = df.fillna(0)
    # Ensure column order & missing features
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    return df[FEATURES]

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "json body required"}), 400
    rec = prepare_single(payload)
    pred = MODEL.predict(rec, num_iteration=MODEL.best_iteration)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
