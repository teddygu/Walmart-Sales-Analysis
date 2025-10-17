"""
train_model.py
Trains a LightGBM model on the processed data and saves model + metrics.
"""

import pandas as pd
import lightgbm as lgb
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'store_code', 'dept_code', 'year', 'month', 'weekofyear',
    'dayofweek', 'day', 'us_holiday', 'IsHoliday',
    'days_from_start', 'lag_1', 'lag_2', 'lag_3', 'lag_52',
    'rolling_4', 'rolling_8'
]
TARGET = 'Weekly_Sales'

def load_data():
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    return train, val

def train_lgb(train_df, val_df):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'n_estimators': 1000,
        'verbosity': -1,
        'random_state': 42
    }

    train_X = train_df[FEATURES]
    train_y = train_df[TARGET]
    val_X = val_df[FEATURES]
    val_y = val_df[TARGET]

    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_val = lgb.Dataset(val_X, val_y, reference=lgb_train)

    model = lgb.train(params,
                      lgb_train,
                      valid_sets=[lgb_train, lgb_val],
                      early_stopping_rounds=50,
                      verbose_eval=50)

    return model

def evaluate(model, val_df):
    preds = model.predict(val_df[FEATURES], num_iteration=model.best_iteration)
    rmse = mean_squared_error(val_df[TARGET], preds, squared=False)
    mae = mean_absolute_error(val_df[TARGET], preds)
    mape = np.mean(np.abs((val_df[TARGET] - preds) / (val_df[TARGET] + 1e-9))) * 100
    return {'rmse': rmse, 'mae': mae, 'mape': mape}, preds

def main():
    train_df, val_df = load_data()
    model = train_lgb(train_df, val_df)
    metrics, preds = evaluate(model, val_df)
    print("Validation metrics:", metrics)
    # Save model and feature list
    joblib.dump(model, MODEL_DIR / "lgb_model.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_list.joblib")
    # Save sample preds to inspect
    sample = val_df[['Store','Dept','Date','Weekly_Sales']].copy()
    sample['pred'] = preds
    sample.to_csv(MODEL_DIR / "val_predictions.csv", index=False)
    print("Saved model and predictions to", MODEL_DIR)

if __name__ == "__main__":
    main()
