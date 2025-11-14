import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import mlflow
import os
from dotenv import load_dotenv

# 1. Load Secrets & Connect to MLflow
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

print("📂 Loading All Datasets (Train, Oil, Stores, Holidays)...")
try:
    # Use latin1 encoding for train.csv due to special characters
    df_train = pd.read_csv("data/train.csv", encoding='latin1', low_memory=False)
    
    # Load supplementary data
    df_oil = pd.read_csv("data/oil.csv")
    df_stores = pd.read_csv("data/stores.csv")
    df_holidays = pd.read_csv("data/holidays_events.csv")
    
    print("  ✅ All CSVs loaded successfully.")
except Exception as e:
    print(f"❌ Error loading data. Did you run the Kaggle download?\n{e}")
    exit()

# 2. Advanced Preprocessing & Feature Engineering
print("⚙️ Merging datasets and engineering new features...")

# --- Convert all date columns ---
df_train['date'] = pd.to_datetime(df_train['date'])
df_oil['date'] = pd.to_datetime(df_oil['date'])
df_holidays['date'] = pd.to_datetime(df_holidays['date'])

# --- 1. Engineer Oil Prices ---
# Oil data is missing weekends. We forward-fill (ffill) to apply Friday's price to Sat/Sun.
df_oil = df_oil.set_index('date').resample('D').ffill().reset_index()
# Merge oil data
df = pd.merge(df_train, df_oil, on='date', how='left')
# Backfill any remaining NaNs (e.g., at the very start of the dataset)
df['dcoilwtico'] = df['dcoilwtico'].fillna(method='bfill')

# --- 2. Engineer Store Data ---
df = pd.merge(df, df_stores, on='store_nbr', how='left')

# --- 3. Engineer Holiday Data ---
# We only care about actual holidays, not transferred ones
df_holidays = df_holidays[df_holidays['transferred'] == False]
# Create a simple 'is_holiday' flag
df_holidays['is_holiday'] = 1
# Merge
df = pd.merge(df, df_holidays[['date', 'is_holiday']], on='date', how='left')
# Fill non-holidays with 0
df['is_holiday'] = df['is_holiday'].fillna(0)

# --- 4. Engineer Time Features ---
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_month'] = df['date'].dt.day

# --- 5. Encode Categorical Features ---
# XGBoost needs numbers, not strings like "Quito" or "GROCERY I"
encoders = {}
for col in ['family', 'city', 'state', 'type']:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    encoders[col] = le # In a real project, you would save these encoders

# 3. Define Features & Split Data (Time-Series Aware)
print("🎯 Defining feature set and splitting data...")

# Our model is now much more powerful!
FEATURES = [
    'store_nbr', 'family_encoded', 'onpromotion',
    'dcoilwtico', 'is_holiday', 
    'city_encoded', 'state_encoded', 'type_encoded', 
    'day_of_week', 'month', 'year', 'day_of_month'
]
TARGET = 'sales'

# --- Time-Series Split (Crucial!) ---
# We must validate on the "future", not on a random sample.
# We will train on data up to 2016 and validate on 2017.
train_data = df[df['date'] < '2017-01-01']
test_data = df[df['date'] >= '2017-01-01']

X_train = train_data[FEATURES]
y_train = train_data[TARGET]
X_test = test_data[FEATURES]
y_test = test_data[TARGET]

print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_test):,}")

# 4. Train "v2" Model
print("🚀 Starting Advanced Training (v2 Model)...")
# Set a new experiment name for this advanced model
mlflow.set_experiment("Retail_Sales_Prediction_v2")

with mlflow.start_run():
    # A more powerful model configuration
    model = xgb.XGBRegressor(
        n_estimators=1000,         # More trees
        learning_rate=0.05,
        max_depth=10,              # Deeper trees
        early_stopping_rounds=50,  # Stops training if it doesn't improve
        n_jobs=-1                  # Use all CPU cores
    )
    
    # Train with early stopping
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)
    
    # Predict
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"✅ Advanced Model Trained! MAE: {mae:.4f}")
    
    # Log Metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_params({
        "n_estimators": 1000,
        "max_depth": 10,
        "features_count": len(FEATURES)
    })
    
    # Save the new model with a "v2" name
    print("📦 Saving v2 model artifact...")
    model.save_model("sales_model_v2.json")
    mlflow.log_artifact("sales_model_v2.json")
    
    print("✨ Advanced Model (sales_model_v2.json) uploaded successfully!")