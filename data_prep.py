"""
data_prep.py
Load raw Walmart-like CSV, engineer features, and save processed train/test CSVs.
Assumes raw CSV has columns: Store, Dept, Date, Weekly_Sales, IsHoliday (bool/int or Y/N)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import holidays
import joblib

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "walmart_raw.csv"   # point this at your CSV
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

US_HOLIDAYS = holidays.UnitedStates()

def load_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

def basic_clean(df):
    # Standardize column names
    df = df.rename(columns=lambda x: x.strip())
    # Ensure IsHoliday is boolean
    if df['IsHoliday'].dtype == object:
        df['IsHoliday'] = df['IsHoliday'].map({'Y': True, 'N': False}).fillna(False)
    df['IsHoliday'] = df['IsHoliday'].astype(bool)
    # Drop rows missing target
    df = df.dropna(subset=['Weekly_Sales'])
    return df

def make_date_features(df):
    df = df.sort_values(['Store', 'Dept', 'Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['dayofweek'] = df['Date'].dt.weekday
    df['day'] = df['Date'].dt.day
    # is holiday from calendar (in case some holidays not flagged)
    df['us_holiday'] = df['Date'].apply(lambda d: d in US_HOLIDAYS)
    # relative time
    df['days_from_start'] = (df['Date'] - df['Date'].min()).dt.days
    return df

def lag_features(df, lags=[1,2,3,52]):
    # Create group-based lags for Weekly_Sales per Store+Dept
    df = df.sort_values(['Store','Dept','Date'])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(lag)
    # moving average features
    df['rolling_4'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(4).mean()
    df['rolling_8'] = df.groupby(['Store','Dept'])['Weekly_Sales'].shift(1).rolling(8).mean()
    return df

def encode_categoricals(df):
    # Simple label encoding for Store and Dept
    df['Store'] = df['Store'].astype(str)
    df['Dept'] = df['Dept'].astype(str)
    df['store_code'] = df['Store'].astype('category').cat.codes
    df['dept_code'] = df['Dept'].astype('category').cat.codes
    # Save encoders (categories) for later use in inference
    encoders = {
        'store_categories': df['Store'].astype('category').cat.categories.tolist(),
        'dept_categories': df['Dept'].astype('category').cat.categories.tolist()
    }
    joblib.dump(encoders, PROCESSED_DIR / "encoders.joblib")
    return df

def split_time_series(df, val_weeks=12):
    # Hold out last val_weeks as validation (per group)
    last_date = df['Date'].max()
    cutoff = last_date - pd.Timedelta(weeks=val_weeks)
    train = df[df['Date'] <= cutoff].copy()
    val = df[df['Date'] > cutoff].copy()
    return train, val

def main():
    df = load_data(RAW_CSV)
    df = basic_clean(df)
    df = make_date_features(df)
    df = lag_features(df)
    df = encode_categoricals(df)
    # drop rows with NaN lag features (first weeks)
    df = df.dropna(subset=[c for c in df.columns if c.startswith('lag_')] + ['rolling_4','rolling_8'])
    train, val = split_time_series(df)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    print("Saved processed train/val to", PROCESSED_DIR)

if __name__ == "__main__":
    main()
