import streamlit as st
import pandas as pd
import xgboost as xgb
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import time

# 1. Load Config
load_dotenv()
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
except Exception as e:
    st.error(f"Failed to connect to Redis. Check .env variables.\n{e}")
    st.stop()

# Load the Model we just trained
try:
    model = xgb.XGBRegressor()
    model.load_model("sales_model.json")
except Exception as e:
    st.error(f"Failed to load model 'sales_model.json'. Did you run train.py?\n{e}")
    st.stop()

# --- UI LAYOUT ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

# Centered Title
st.markdown(
    """
    <style>
    /* --- THEME FIX --- */
    /* We use Streamlit's built-in theme variables so it works in Dark & Light mode */
    .stMetric {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Target the label inside the metric */
    .stMetric [data-testid="stMetricLabel"] {
        color: var(--secondary-text-color);
    }

    /* Target the value inside the metric */
    .stMetric [data-testid="stMetricValue"] {
        color: var(--primary-text-color);
    }
    /* --- END OF FIX --- */
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🛒 Retail Demand Live Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>End-to-End MLOps Pipeline: <b>Redis Stream (Ingestion)</b> → <b>Redis Feature Store (Live)</b> → <b>XGBoost (Inference)</b></p>", unsafe_allow_html=True)
st.divider()

# Create two columns with more defined spacing
col1, col2 = st.columns([0.45, 0.55]) 

# --- COLUMN 1: LIVE DATA (REDIS) ---
with col1:
    with st.container(border=True, height=600):
        st.subheader("📡 Real-Time Feature Store")
        st.caption("Reading live rolling-window metrics from Upstash Redis.")
        
        # User selects an item family to monitor
        family = st.selectbox("Select Item Family to Monitor", 
                              ["POULTRY", "DAIRY", "MEATS", "BEVERAGES", "PRODUCE", "CLEANING", "GROCERY I"])
        
        # Fetch data from Redis
        redis_key = f"feature:sales_volume:{family}"
        
        val = redis.get(redis_key)
        
        # Refresh button
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            val = redis.get(redis_key)
            
        current_volume = float(val) if val else 0.0
        
        # Theme-aware metric
        st.metric(label=f"Live Aggregate Sales ({family})", value=f"${current_volume:,.2f}")
        
        # st.info("Run `producer.py` and `feature_store.py` in the background to see this number update!")

        # --- ARCHITECTURE DIAGRAM (Code-based) ---
        st.markdown("### System Architecture")
        try:
            st.graphviz_chart('''
                digraph {
                    rankdir=LR
                    node [shape=box style="filled,rounded" fillcolor="#f0f2f6" fontname="Helvetica" color="#aaaaaa"]
                    edge [fontname="Helvetica" color="#aaaaaa" fontcolor="#aaaaaa"]
                    
                    PROD [label="Producer" shape=ellipse fillcolor="#e3f2fd"]
                    STREAM [label="Redis Stream" fillcolor="#ffebee"]
                    PROC [label="Processor" shape=ellipse fillcolor="#e3f2fd"]
                    STORE [label="Feature Store" shape=cylinder fillcolor="#fff3e0"]
                    DASH [label="Dashboard" fillcolor="#e8f5e9"]

                    PROD -> STREAM
                    STREAM -> PROC 
                    PROC -> STORE 
                    STORE -> DASH 
                }
            ''')
        except Exception:
            st.warning("Graphviz not installed. Skipping diagram.")


# --- COLUMN 2: INFERENCE (MODEL) ---
with col2:
    with st.container(border=True, height=600):
        st.subheader("🤖 Run On-Demand Prediction")
        st.caption("Predict sales for a future scenario using the trained XGBoost model.")
        
        # Inputs for the model
        c1, c2 = st.columns(2)
        with c1:
            store_nbr = st.slider("Store Number", 1, 50, 1)
            month = st.slider("Month", 1, 12, 1)
        with c2:
            day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
            is_promo = st.checkbox("Item is on Promotion?")
        
        if st.button("🔮 Predict Sales", use_container_width=True, type="primary"):
            # Prepare input dataframe
            input_data = pd.DataFrame({
                'store_nbr': [store_nbr],
                'onpromotion': [1 if is_promo else 0],
                'day_of_week': [day_of_week],
                'month': [month]
            })
            
            # Predict
            pred = model.predict(input_data)[0]
            
            # Theme-aware metric
            st.metric(label="Predicted Sales Volume", value=f"{pred:.2f} units")
            
            # Visual check logic
            if pred > 500:
                st.warning("⚠️ High Demand Alert! Recommend stocking up inventory.")
            else:
                st.success("✅ Demand is within normal range.")