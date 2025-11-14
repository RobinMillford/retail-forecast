import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import mlflow
import os
import joblib
import json
from dotenv import load_dotenv
from upstash_redis import Redis
from prophet import Prophet

# 1. LOAD CONFIG
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

TRAINING_BUFFER_KEY = "training_data_buffer"
redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))

print("📂 Loading Data...")
try:
    df_history = pd.read_csv("data/train.csv", encoding='latin1', low_memory=False)
    df_oil = pd.read_csv("data/oil.csv")
    df_stores = pd.read_csv("data/stores.csv")
    df_holidays = pd.read_csv("data/holidays_events.csv")
    df_transactions = pd.read_csv("data/transactions.csv")
    print(f"  ✅ Loaded history: {len(df_history):,} rows")
except Exception as e:
    print(f"❌ Error loading CSVs: {e}")
    exit()

# Redis Buffer Logic
df_fresh = pd.DataFrame()
try:
    new_data_raw = redis.lrange(TRAINING_BUFFER_KEY, 0, -1)
    if len(new_data_raw) > 0:
        print(f"  ✅ Found {len(new_data_raw)} fresh records!")
        redis.delete(TRAINING_BUFFER_KEY)
        new_data_json = [json.loads(row) for row in new_data_raw]
        df_fresh = pd.DataFrame(new_data_json)
        df_fresh['date'] = pd.to_datetime(df_fresh['date'])
        df_fresh['store_nbr'] = df_fresh['store_nbr'].astype(int)
        df_fresh['sales'] = df_fresh['sales'].astype(float)
        df_fresh['onpromotion'] = df_fresh['onpromotion'].astype(int)
except:
    pass

df_train = pd.concat([df_history, df_fresh], ignore_index=True)

# Feature Engineering
print("⚙️ Engineering Features...")
df_train['date'] = pd.to_datetime(df_train['date'])
df_oil['date'] = pd.to_datetime(df_oil['date'])
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
df_transactions['date'] = pd.to_datetime(df_transactions['date'])

df_oil = df_oil.set_index('date').resample('D').ffill().reset_index()
df = pd.merge(df_train, df_oil, on='date', how='left')
df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()

df = pd.merge(df, df_stores, on='store_nbr', how='left')
df_holidays = df_holidays[df_holidays['transferred'] == False]
df_holidays['is_holiday'] = 1
df = pd.merge(df, df_holidays[['date', 'is_holiday']], on='date', how='left')
df['is_holiday'] = df['is_holiday'].fillna(0)
df = pd.merge(df, df_transactions, on=['date', 'store_nbr'], how='left')
df['transactions'] = df['transactions'].fillna(0)

df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_month'] = df['date'].dt.day

encoders = {}
for col in ['family', 'city', 'state', 'type']:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save Encoders
joblib.dump(encoders['family'], 'family_encoder.joblib')
joblib.dump(encoders['city'], 'city_encoder.joblib')
joblib.dump(encoders['state'], 'state_encoder.joblib')
joblib.dump(encoders['type'], 'type_encoder.joblib')

# Global Split Date
val_date = '2017-08-01'

mlflow.set_experiment("Retail_Prediction_Combined_v3")

# --- START PARENT RUN ---
print("🚀 Starting MLflow Run...")
with mlflow.start_run(run_name="Nightly_Pipeline_Run") as parent_run:
    
    mlflow.log_param("total_rows", len(df))
    
    # Log Encoders to Parent Run
    mlflow.log_artifact("family_encoder.joblib")
    mlflow.log_artifact("city_encoder.joblib")
    mlflow.log_artifact("state_encoder.joblib")
    mlflow.log_artifact("type_encoder.joblib")

    # ==========================
    # CHILD RUN 1: XGBoost
    # ==========================
    with mlflow.start_run(run_name="XGBoost_Training", nested=True):
        FEATURES = ['store_nbr', 'family_encoded', 'onpromotion', 'transactions', 
                    'dcoilwtico', 'is_holiday', 'city_encoded', 'state_encoded', 
                    'type_encoded', 'day_of_week', 'month', 'year', 'day_of_month']
        TARGET = 'sales'
        
        # Split for XGBoost
        train_data = df[df['date'] < val_date]
        test_data = df[df['date'] >= val_date]
        
        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10, 
                                 early_stopping_rounds=50, n_jobs=-1, random_state=42)
        
        model.fit(train_data[FEATURES], train_data[TARGET], 
                  eval_set=[(test_data[FEATURES], test_data[TARGET])], verbose=False)
        
        preds = model.predict(test_data[FEATURES])
        preds[preds < 0] = 0
        mae = mean_absolute_error(test_data[TARGET], preds)
        
        print(f"  ✅ XGBoost MAE: {mae:.4f}")
        mlflow.log_metric("mae", mae)
        
        model.save_model("best_model_v2.json")
        mlflow.log_artifact("best_model_v2.json")

    # ==========================
    # CHILD RUN 2: Prophet (FIXED METRICS)
    # ==========================
    with mlflow.start_run(run_name="Prophet_Training", nested=True):
        # 1. Prepare Aggregated Data (Daily Total Sales)
        # We aggregate by date to get the total company trend
        df_prophet = df.groupby('date').agg({
            'sales': 'sum',
            'dcoilwtico': 'mean',
            'is_holiday': 'max'
        }).reset_index()
        df_prophet = df_prophet.rename(columns={'date': 'ds', 'sales': 'y'})
        
        # 2. Split for Prophet
        p_train = df_prophet[df_prophet['ds'] < val_date]
        p_test = df_prophet[df_prophet['ds'] >= val_date]
        
        m = Prophet()
        m.add_regressor('dcoilwtico')
        m.add_regressor('is_holiday')
        
        # 3. Fit on Training Set
        m.fit(p_train)
        
        # 4. Evaluate on Test Set
        future_test = p_test[['ds', 'dcoilwtico', 'is_holiday']]
        forecast_test = m.predict(future_test)
        
        # Calculate Metrics
        preds_p = forecast_test['yhat'].values
        actuals_p = p_test['y'].values
        mae_p = mean_absolute_error(actuals_p, preds_p)
        
        print(f"  ✅ Prophet MAE: {mae_p:.4f}")
        mlflow.log_metric("mae", mae_p)
        
        # 5. Refit on Full Data (Optional, but good for the dashboard forecast)
        # For the live dashboard, we want the model to know about the latest data too.
        m_final = Prophet()
        m_final.add_regressor('dcoilwtico')
        m_final.add_regressor('is_holiday')
        m_final.fit(df_prophet) # Fit on ALL data
        
        joblib.dump(m_final, "long_term_forecast.pkl")
        mlflow.log_artifact("long_term_forecast.pkl")

print("✨ Pipeline Complete. Check Dagshub for nested runs.")