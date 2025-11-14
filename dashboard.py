import streamlit as st
import pandas as pd
import xgboost as xgb
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import time
from datetime import datetime

# 1. Load Config & Connect
load_dotenv()
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    redis.ping()
except Exception as e:
    st.error(f"Failed to connect to Redis. Check .env variables.\n{e}")
    st.stop()

# 2. Load Model
@st.cache_resource
def load_xgb_model():
    try:
        model = xgb.XGBRegressor()
        model.load_model("sales_model.json")
        return model
    except Exception as e:
        st.error(f"Failed to load model 'sales_model.json'. Did you run train.py?\n{e}")
        st.stop()
model = load_xgb_model()

# --- UI CONFIG ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

# Custom CSS for metrics
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
st.markdown("<p style='text-align: center; font-size: 18px;'>End-to-End MLOps: <b>Live Ingestion</b> • <b>Feature Store</b> • <b>Inference</b></p>", unsafe_allow_html=True)
st.divider()

# --- MAIN COLUMNS ---
col1, col2 = st.columns([0.45, 0.55]) 

# ==========================================
# COLUMN 1: LIVE DATA (REDIS)
# ==========================================
with col1:
    with st.container(border=True, height=700):
        st.subheader("📡 Live Feature Store")
        st.caption("Real-time aggregation of sales data flowing through Redis Streams.")
        
        # 1. Select Item
        ITEM_FAMILIES = (
            'GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY', 'POULTRY', 
            'MEATS', 'PERSONAL CARE', 'DELI', 'HOME CARE', 'EGGS', 'FROZEN FOODS'
        )
        family = st.selectbox("Select Item Family to Monitor", ITEM_FAMILIES)
        
        # 2. Fetch Data Directly (Fix for the "not updating" bug)
        # We fetch fresh data every time the app reruns (which happens on selection change)
        redis_key = f"feature:sales_volume:{family}"
        val = redis.get(redis_key)
        current_volume = float(val) if val else 0.0
        
        # 3. Refresh Button
        # Clicking this just triggers a rerun, which re-executes the fetch above
        st.button("🔄 Refresh Live Data", use_container_width=True)
        
        # 4. Display Metric
        st.metric(label=f"Live 7-Day Rolling Sales ({family})", value=f"${current_volume:,.2f}")
        
        st.info("The background worker (`live_stream.yml`) updates this value every 5 minutes.")

        st.markdown("---")
        st.markdown("### System Architecture")
        try:
            st.graphviz_chart('''
                digraph {
                    rankdir=LR
                    node [shape=box style="filled,rounded" fillcolor="#f0f2f6" fontname="Helvetica" color="#aaaaaa"]
                    edge [fontname="Helvetica" color="#aaaaaa" fontcolor="#aaaaaa"]
                    
                    subgraph cluster_actions {
                        label = "GitHub Actions"
                        style="filled,rounded"
                        fillcolor="#fafafa"
                        color="#dddddd"
                        PROD [label="Producer" shape=ellipse fillcolor="#e3f2fd"]
                        PROC [label="Processor" shape=ellipse fillcolor="#e3f2fd"]
                    }

                    STREAM [label="Redis Stream" fillcolor="#ffebee"]
                    STORE [label="Redis\nFeature Store" shape=cylinder fillcolor="#fff3e0"]
                    DASH [label="Dashboard" fillcolor="#e8f5e9"]
                    
                    PROD -> STREAM
                    PROC -> STREAM
                    PROC -> STORE
                    DASH -> STORE
                }
            ''')
        except Exception:
            st.warning("Diagram unavailable.")


# ==========================================
# COLUMN 2: INFERENCE (MODEL)
# ==========================================
with col2:
    with st.container(border=True, height=700):
        st.subheader("🔮 Run On-Demand Prediction")
        st.caption("Predict sales for a specific store and date using the XGBoost model.")
        
        # 1. Diverse Store List (One per city to avoid repetition)
        STORE_MAP = {
            "Quito (Main Branch) - Store 1": 1,
            "Santo Domingo - Store 5": 5,
            "Cayambe - Store 11": 11,
            "Latacunga - Store 14": 14,
            "Riobamba - Store 17": 17,
            "Guayaquil (Port) - Store 24": 24,
            "Machala - Store 41": 41,
            "Cuenca - Store 37": 37,
            "Loja - Store 50": 50,
            "Manta (Coastal) - Store 53": 53,
            "El Carmen - Store 54": 54
        }
        store_name = st.selectbox("Select Store Location", STORE_MAP.keys())
        
        # 2. Date
        col_date, col_promo = st.columns(2)
        with col_date:
            prediction_date = st.date_input("Select Date", datetime.now())
        with col_promo:
            st.write("") # Spacer
            st.write("") 
            is_promo = st.toggle("APPLY PROMOTION?", value=False)
        
        st.divider()

        # Feature Prep
        month_feature = prediction_date.month
        day_of_week_feature = prediction_date.weekday()
        onpromotion_feature = 1 if is_promo else 0
        store_nbr_feature = STORE_MAP[store_name]

        # Expander for clarity
        with st.expander("View Model Input Vector"):
            st.code(f"""
            {{
                "store_nbr": {store_nbr_feature},
                "onpromotion": {onpromotion_feature},
                "month": {month_feature},
                "day_of_week": {day_of_week_feature}
            }}
            """, language="json")

        if st.button("🔮 Calculate Prediction", use_container_width=True, type="primary"):
            
            # Create DataFrame
            input_data = pd.DataFrame({
                'store_nbr': [store_nbr_feature],
                'onpromotion': [onpromotion_feature],
                'day_of_week': [day_of_week_feature],
                'month': [month_feature]
            })
            
            # Get Prediction
            pred = model.predict(input_data)[0]
            
            # Display
            st.subheader(f"Forecast for {prediction_date.strftime('%B %d, %Y')}")
            
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                st.metric(label="Predicted Unit Sales", value=f"{pred:.2f}")
            
            with col_res2:
                if pred > 400:
                    st.error("🔥 HIGH DEMAND")
                elif pred > 100:
                    st.warning("⚠️ MODERATE")
                else:
                    st.success("✅ NORMAL")