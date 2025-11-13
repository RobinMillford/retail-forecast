import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import os
from dotenv import load_dotenv

# 1. Load Secrets
load_dotenv()

# Set up MLflow to talk to the Cloud
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

print("📂 Loading Data...")
# Check if file exists
if not os.path.exists("data/train.csv"):
    print("❌ Error: data/train.csv not found.")
    exit()

df = pd.read_csv("data/train.csv")

# 2. Preprocessing
print("🧹 Preprocessing...")
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

train_df = df[df['year'] < 2017]

features = ['store_nbr', 'onpromotion', 'day_of_week', 'month']
target = 'sales'

X = train_df[features]
y = train_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Start MLflow Run
print("🚀 Starting Training...")
mlflow.set_experiment("Retail_Sales_Prediction")

with mlflow.start_run():
    # Define Model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"✅ Model Trained! MAE: {mae}")
    
    # Log Metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mae", mae)
    
    # --- THE FIX IS HERE ---
    # Instead of log_model (which fails), we save manually and upload
    print("📦 Saving model artifact...")
    model.save_model("sales_model.json")
    mlflow.log_artifact("sales_model.json")
    
    print("✨ Model file (sales_model.json) uploaded to Dagshub successfully!")