import streamlit as st
import pandas as pd
import xgboost as xgb
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import joblib 
import plotly.graph_objects as go
import numpy as np

# 1. Load Config & Connect
load_dotenv()
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    redis.ping()
except Exception as e:
    st.error(f"Failed to connect to Redis. Check .env variables.\n{e}")
    st.stop()

# 2. Load ALL Models & Encoders
@st.cache_resource
def load_assets():
    """Loads all models (XGBoost, Prophet) and encoders."""
    try:
        model_xgb = xgb.XGBRegressor()
        model_xgb.load_model("best_model_v2.json")
        
        model_prophet = joblib.load("long_term_forecast.pkl")
        
        encoders = {
            "family": joblib.load("family_encoder.joblib"),
            "city": joblib.load("city_encoder.joblib"),
            "state": joblib.load("state_encoder.joblib"),
            "type": joblib.load("type_encoder.joblib")
        }
        return model_xgb, model_prophet, encoders
    except Exception as e:
        st.error(f"Failed to load model assets. Did you run the nightly training?\n{e}")
        st.stop()
        
model_xgb, model_prophet, encoders = load_assets()

# --- COMPLETE STORE DATABASE ---
STORE_DB = {
    1: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    2: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    3: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    4: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    5: {'city': 'Santo Domingo', 'state': 'Santo Domingo de los Tsachilas', 'type': 'D'},
    6: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    7: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    8: {'city': 'Quito', 'state': 'Pichincha', 'type': 'D'},
    9: {'city': 'Quito', 'state': 'Pichincha', 'type': 'B'},
    10: {'city': 'Quito', 'state': 'Pichincha', 'type': 'C'},
    11: {'city': 'Cayambe', 'state': 'Pichincha', 'type': 'B'},
    12: {'city': 'Latacunga', 'state': 'Cotopaxi', 'type': 'C'},
    13: {'city': 'Latacunga', 'state': 'Cotopaxi', 'type': 'C'},
    14: {'city': 'Riobamba', 'state': 'Chimborazo', 'type': 'C'},
    15: {'city': 'Ibarra', 'state': 'Imbabura', 'type': 'C'},
    16: {'city': 'Santo Domingo', 'state': 'Santo Domingo de los Tsachilas', 'type': 'C'},
    17: {'city': 'Quito', 'state': 'Pichincha', 'type': 'C'},
    18: {'city': 'Quito', 'state': 'Pichincha', 'type': 'B'},
    19: {'city': 'Guaranda', 'state': 'Bolivar', 'type': 'C'},
    20: {'city': 'Quito', 'state': 'Pichincha', 'type': 'B'},
    21: {'city': 'Santo Domingo', 'state': 'Santo Domingo de los Tsachilas', 'type': 'B'},
    22: {'city': 'Puyo', 'state': 'Pastaza', 'type': 'C'},
    23: {'city': 'Ambato', 'state': 'Tungurahua', 'type': 'D'},
    24: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'D'},
    25: {'city': 'Salinas', 'state': 'Santa Elena', 'type': 'D'},
    26: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'D'},
    27: {'city': 'Daule', 'state': 'Guayas', 'type': 'D'},
    28: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'E'},
    29: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'E'},
    30: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'C'},
    31: {'city': 'Babahoyo', 'state': 'Los Rios', 'type': 'B'},
    32: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'C'},
    33: {'city': 'Quevedo', 'state': 'Los Rios', 'type': 'C'},
    34: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'B'},
    35: {'city': 'Playas', 'state': 'Guayas', 'type': 'C'},
    36: {'city': 'Libertad', 'state': 'Guayas', 'type': 'E'},
    37: {'city': 'Cuenca', 'state': 'Azuay', 'type': 'D'},
    38: {'city': 'Loja', 'state': 'Loja', 'type': 'D'},
    39: {'city': 'Cuenca', 'state': 'Azuay', 'type': 'D'},
    40: {'city': 'Machala', 'state': 'El Oro', 'type': 'C'},
    41: {'city': 'Machala', 'state': 'El Oro', 'type': 'D'},
    42: {'city': 'Cuenca', 'state': 'Azuay', 'type': 'D'},
    43: {'city': 'Esmeraldas', 'state': 'Esmeraldas', 'type': 'E'},
    44: {'city': 'Quito', 'state': 'Pichincha', 'type': 'A'},
    45: {'city': 'Quito', 'state': 'Pichincha', 'type': 'A'},
    46: {'city': 'Quito', 'state': 'Pichincha', 'type': 'A'},
    47: {'city': 'Quito', 'state': 'Pichincha', 'type': 'A'},
    48: {'city': 'Quito', 'state': 'Pichincha', 'type': 'A'},
    49: {'city': 'Quito', 'state': 'Pichincha', 'type': 'A'},
    50: {'city': 'Ambato', 'state': 'Tungurahua', 'type': 'A'},
    51: {'city': 'Guayaquil', 'state': 'Guayas', 'type': 'A'},
    52: {'city': 'Manta', 'state': 'Manabi', 'type': 'A'},
    53: {'city': 'Manta', 'state': 'Manabi', 'type': 'D'},
    54: {'city': 'El Carmen', 'state': 'Manabi', 'type': 'C'}
}

# --- UI CONFIG ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .stMetric {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
    }
    .stMetric [data-testid="stMetricLabel"] {
        color: var(--secondary-text-color);
    }
    .stMetric [data-testid="stMetricValue"] {
        color: var(--primary-text-color);
    }
    </style>
    """, unsafe_allow_html=True
)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🛒 Retail Demand Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Using a multi-model (XGBoost + Prophet) MLOps pipeline.</p>", unsafe_allow_html=True)
st.divider()

# --- MAIN COLUMNS ---
col1, col2 = st.columns([0.45, 0.55]) 

# ==========================================
# COLUMN 1: LIVE DATA & ARCHITECTURE
# ==========================================
with col1:
    with st.container(border=True, height=800):
        st.subheader("📡 Live Feature Store")
        st.caption("Aggregated sales data from Redis (updated every 5 mins).")
        
        time_window = st.radio(
            "Select Time Window",
            ("Today", "This Week", "This Month"),
            horizontal=True,
        )
        ITEM_FAMILIES = ('GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY')
        family = st.selectbox("Select Item Family", ITEM_FAMILIES)
        
        today = datetime.now()
        if time_window == "Today":
            date_key = today.strftime('%Y-%m-%d')
            redis_key = f"feature:sales_daily:{family}:{date_key}"
            label = f"Live Sales Today ({family})"
        elif time_window == "This Week":
            date_key = today.strftime('%Y-W%U')
            redis_key = f"feature:sales_weekly:{family}:{date_key}"
            label = f"Live Sales This Week ({family})"
        else:
            date_key = today.strftime('%Y-%m')
            redis_key = f"feature:sales_monthly:{family}:{date_key}"
            label = f"Live Sales This Month ({family})"
        
        # --- FIX IS HERE ---
        # 1. We read the value directly (no session state)
        # 2. The 'Refresh' button just re-runs the script, which re-runs this line
        val = redis.get(redis_key)
        
        st.button("🔄 Refresh Live Data", use_container_width=True)
        
        # 3. We use the 'val' we just defined
        current_volume = float(val) if val else 0.0
        st.metric(label=label, value=f"${current_volume:,.2f}")
        # --- END OF FIX ---

        st.markdown("---")
        st.markdown("### System Architecture")
        try:
            st.graphviz_chart('''
                digraph {
                    rankdir=LR
                    node [shape=box style="filled,rounded" fillcolor="#f0f2f6" fontname="Helvetica" color="#aaaaaa"]
                    edge [fontname="Helvetica" color="#aaaaaa" fontcolor="#aaaaaa"]
                    
                    subgraph cluster_actions {
                        label = "GitHub Actions (Scheduler)"
                        style="filled,rounded"
                        fillcolor="#fafafa"
                        color="#dddddd"
                        KAGGLE [label="Kaggle API" shape=cylinder fillcolor="#e0f7fa"]
                        PROD [label="Producer" shape=ellipse fillcolor="#e3f2fd"]
                        PROC [label="Processor" shape=ellipse fillcolor="#e3f2fd"]
                        TRAIN [label="Nightly Trainer" shape=ellipse fillcolor="#e3f2fd"]
                    }
                    STORE [label="Redis\nFeature Store" shape=cylinder fillcolor="#fff3e0"]
                    DASH [label="Dashboard" fillcolor="#e8f5e9"]
                    MLFLOW [label="MLflow" shape=cylinder fillcolor="#ede7f6"]
                    
                    KAGGLE -> PROD
                    KAGGLE -> TRAIN
                    PROD -> STORE [label="sends live data"]
                    PROC -> STORE [label="aggregates data"]
                    TRAIN -> MLFLOW [label="logs metrics"]
                    TRAIN -> DASH [label="deploys models"]
                    DASH -> STORE [label="reads live data"]
                }
            ''')
        except Exception:
            st.warning("Diagram unavailable.")

# ==========================================
# COLUMN 2: FORECASTING
# ==========================================
with col2:
    with st.container(border=True, height=800):
        
        tab1, tab2 = st.tabs(["🔮 Single-Day Prediction (XGBoost)", "📈 Long-Term Forecast (Prophet)"])

        # --- TAB 1: XGBOOST (HIGH ACCURACY) ---
        with tab1:
            st.subheader("Predict Specific Item Sales")
            st.caption("Powered by XGBoost (v2). Calculates precise sales for any store/item combo.")
            
            # Use full list of stores
            store_options = [f"Store {k} - {v['city']} ({v['state']})" for k, v in STORE_DB.items()]
            store_key = st.selectbox("Select Store Location", store_options)
            
            # Use full list of items
            family_key = st.selectbox("Select Item Family", encoders['family'].classes_)
            
            col_d, col_p = st.columns(2)
            prediction_date = col_d.date_input("Select Date", datetime.now())
            is_promo = col_p.toggle("Apply Promotion?", value=False)
            
            if st.button("Calculate Single-Day Prediction", use_container_width=True, type="primary"):
                # 1. Get user inputs
                selected_store_id = int(store_key.split(' ')[1])
                store_meta = STORE_DB[selected_store_id]
                
                # 2. Get time features
                month, day_of_week, year, day_of_month = (
                    prediction_date.month, prediction_date.weekday(), 
                    prediction_date.year, prediction_date.day
                )
                
                # 3. Simulate other features (using simple defaults)
                default_oil = 45.0
                default_transactions = 1500
                default_holiday = 0
                
                try:
                    # 4. Use ENCODERS to transform text -> numbers
                    family_encoded = encoders['family'].transform([family_key])[0]
                    city_encoded = encoders['city'].transform([store_meta['city']])[0]
                    state_encoded = encoders['state'].transform([store_meta['state']])[0]
                    type_encoded = encoders['type'].transform([store_meta['type']])[0]

                    # 5. Build the feature vector
                    input_data = pd.DataFrame([{
                        'store_nbr': selected_store_id, 'family_encoded': family_encoded,
                        'onpromotion': 1 if is_promo else 0, 'transactions': default_transactions,
                        'dcoilwtico': default_oil, 'is_holiday': default_holiday,
                        'city_encoded': city_encoded, 'state_encoded': state_encoded,
                        'type_encoded': type_encoded, 'day_of_week': day_of_week,
                        'month': month, 'year': year, 'day_of_month': day_of_month
                    }])
                    
                    # 6. Predict
                    pred = model_xgb.predict(input_data)[0]
                    pred = max(0, pred) # No negative sales
                    
                    st.metric(f"Predicted Sales for {family_key}", f"{pred:.2f} units")

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        # --- TAB 2: PROPHET (TRENDS) ---
        with tab2:
            st.subheader("Forecast Total Business Trend")
            st.caption("Powered by Prophet. Use this for high-level strategic planning.")
            
            forecast_days = st.slider("Select Forecast Horizon (Days)", 7, 90, 30)
            
            if st.button("Generate Long-Term Forecast", use_container_width=True, type="primary"):
                with st.spinner("Calculating future trends..."):
                    # 1. Create a "future" dataframe
                    future_df = model_prophet.make_future_dataframe(periods=forecast_days)
                    
                    # 2. Add regressor values for the future
                    future_df['dcoilwtico'] = 45.0 # Use a default
                    future_df['is_holiday'] = 0    # Assume no holidays
                    
                    # 3. Predict
                    forecast = model_prophet.predict(future_df)
                    
                    # 4. Plot
                    fig = go.Figure()
                    
                    # Add Uncertainty Band (yhat_upper, yhat_lower)
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_upper'],
                        mode='lines', line=dict(color='rgba(173, 216, 230, 0)'), # Invisible line
                        name='Upper Bound', showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat_lower'],
                        mode='lines', line=dict(color='rgba(173, 216, 230, 0)'), # Invisible line
                        fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)',
                        name='Uncertainty'
                    ))
                    
                    # Add Trend Line
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], y=forecast['yhat'],
                        mode='lines', line=dict(color='blue', width=3),
                        name='Forecast'
                    ))
                    
                    # Add last 30 days of "history"
                    fig.add_trace(go.Scatter(
                        x=model_prophet.history['ds'].iloc[-30:], 
                        y=model_prophet.history['y'].iloc[-30:],
                        mode='lines', line=dict(color='gray'),
                        name='Historical'
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)