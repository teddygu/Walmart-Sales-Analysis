# Walmart-Sales-Analysis-
Walmart Sales Forecasting using Machine Learning

Accurately forecast Walmart’s weekly sales using historical data, feature engineering, and gradient-boosted machine learning models.
This end-to-end project includes data preprocessing, model training, evaluation, an API for live predictions, and a Jupyter-based analytics report.

Features

Feature Engineering:

  -Time-based features (year, month, week, day)
  
  -Lag and rolling average sales
  
  -Holiday detection (via holidays library)
  
  -Store and department encoding

Machine Learning:
  -LightGBM regression model

  -Time-series aware train/validation split
  
  -RMSE, MAE, and MAPE evaluation metrics

Visualization & Reporting:
  -Jupyter notebook for EDA and feature importance
  
  -Auto-generated PDF summary report (via reportlab)
  
  -Correlation heatmaps and seasonal plots
  

Deployment
  -Flask REST API for predictions
  -Ready for Docker or cloud deployment

Technologies Used
| Category         | Tools                  |
| ---------------- | ---------------------- |
| Language         | Python 3.10+           |
| Data Processing  | pandas, numpy          |
| Machine Learning | LightGBM, scikit-learn |
| Visualization    | matplotlib, seaborn    |
| Reporting        | reportlab              |
| API              | Flask                  |
| Version Control  | Git & GitHub           |

Usage
1. Preprocess Data:

   
   			python data_prep.py

     -Generates cleaned and feature-rich train/val datasets in data/processed/.

2. Train Model:
   
   			python train_model.py
   
   -Trains a LightGBM model and saves:

      1.models/lgb_model.joblib

      2.Validation metrics (RMSE, MAE, MAPE)

      3.models/val_predictions.csv

4. Explore Data & Generate Report

Launch Jupyter Notebook:

    jupyter notebook walmart_sales_eda.ipynb


Or let the notebook auto-generate a PDF report in models/Walmart_Sales_Report_YYYYMMDD.pdf.

4. Run the Prediction API
  
  
   		python predict_api.py


Then send a test request:

curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"Store":"1","Dept":"1","Date":"2012-12-07","IsHoliday":false}'


Example EDA Output

1. Total Weekly Sales Over Time


Shows clear seasonality and spikes during major holidays.

2. Monthly and Holiday Effects


Reveals cyclical patterns and increased sales during November–December.

3. Feature Importance
   
Top features include lag_1, rolling_4, and IsHoliday.

4. Auto-generated Report

   
A complete PDF summary is saved to /models, including all visuals and top features.

Evaluation Metrics 
| Metric   | Description                                                   |
| -------- | ------------------------------------------------------------- |
| RMSE | Root Mean Squared Error – measures prediction error magnitude |
| MAE  | Mean Absolute Error – average absolute deviation              |
| MAPE | Mean Absolute Percentage Error – relative accuracy measure    |

Future Improvements:


-Integrate SHAP explainability for model interpretation 

-add weather promotional data as external regressors 

-deploy api via FastAPI or Docker container 

-Build interactive Plotly dashboard for live analysis


