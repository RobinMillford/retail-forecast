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
        model.load_model("sales_model_v2.json")
        return model
    except Exception as e:
        st.error(f"Failed to load model 'sales_model_v2.json'. Did you run train.py?\n{e}")
        st.stop()
model = load_xgb_model()

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
st.markdown("<p style='text-align: center; font-size: 18px;'>End-to-End MLOps: <b>Live Ingestion</b> • <b>Time-Bucketed Feature Store</b> • <b>Inference</b></p>", unsafe_allow_html=True)
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
        
        # --- NEW: Time Window Selector ---
        time_window = st.radio(
            "Select Time Window",
            ("Today", "This Week", "This Month"),
            horizontal=True,
        )
        
        ITEM_FAMILIES = (
            'GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY', 'POULTRY', 
            'MEATS', 'PERSONAL CARE', 'DELI', 'HOME CARE', 'EGGS', 'FROZEN FOODS'
        )
        family = st.selectbox("Select Item Family to Monitor", ITEM_FAMILIES)
        
        # Get today's date info
        today = datetime.now()
        
        # Determine which key to query based on selection
        if time_window == "Today":
            date_key = today.strftime('%Y-%m-%d')
            redis_key = f"feature:sales_daily:{family}:{date_key}"
            label = f"Live Sales Today ({family})"
        elif time_window == "This Week":
            date_key = today.strftime('%Y-W%U') # %U = Week of year (Sunday as first day)
            redis_key = f"feature:sales_weekly:{family}:{date_key}"
            label = f"Live Sales This Week ({family})"
        else: # This Month
            date_key = today.strftime('%Y-%m')
            redis_key = f"feature:sales_monthly:{family}:{date_key}"
            label = f"Live Sales This Month ({family})"

        
        val = redis.get(redis_key)
        current_volume = float(val) if val else 0.0
        
        st.button("🔄 Refresh Live Data", use_container_width=True)
        
        st.metric(label=label, value=f"${current_volume:,.2f}")
        
        st.info("The background worker (`live_stream.yml`) updates these values every 5 minutes.")

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
                    STORE [label="Redis\nTime-Bucketed Store" shape=cylinder fillcolor="#fff3e0"]
                    DASH [label="Dashboard" fillcolor="#e8f5e9"]
                    
                    PROD -> STREAM
                    PROC -> STREAM
                    PROC -> STORE [label="updates 3 keys (Day, Week, Month)"]
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
        
        col_date, col_promo = st.columns(2)
        with col_date:
            prediction_date = st.date_input("Select Date", datetime.now())
        with col_promo:
            st.write("") 
            st.write("") 
            is_promo = st.toggle("APPLY PROMOTION?", value=False)
        
        st.divider()

        # Feature Prep
        month_feature = prediction_date.month
        day_of_week_feature = prediction_date.weekday()
        day_of_month = prediction_date.day
        year = prediction_date.year
        onpromotion_feature = 1 if is_promo else 0
        store_nbr_feature = STORE_MAP[store_name]
        
        # Defaults for missing UI inputs (to prevent model crash)
        DEFAULT_OIL = 45.0
        DEFAULT_HOLIDAY = 0
        DEFAULT_FAMILY = 12 
        DEFAULT_CITY = 18   
        DEFAULT_STATE = 12  
        DEFAULT_TYPE = 3    
        
        with st.expander("View Model Input Vector"):
            st.json({
                "store_nbr": store_nbr_feature,
                "family_encoded": DEFAULT_FAMILY,
                "onpromotion": onpromotion_feature,
                "dcoilwtico": DEFAULT_OIL,
                "is_holiday": DEFAULT_HOLIDAY,
                "city_encoded": DEFAULT_CITY,
                "state_encoded": DEFAULT_STATE,
                "type_encoded": DEFAULT_TYPE,
                "day_of_week": day_of_week_feature,
                "month": month_feature,
                "year": year,
                "day_of_month": day_of_month
            })

        if st.button("🔮 Calculate Prediction", use_container_width=True, type="primary"):
            
            input_data = pd.DataFrame({
                'store_nbr': [store_nbr_feature],
                'family_encoded': [DEFAULT_FAMILY],
                'onpromotion': [onpromotion_feature],
                'dcoilwtico': [DEFAULT_OIL],
                'is_holiday': [DEFAULT_HOLIDAY],
                'city_encoded': [DEFAULT_CITY],
                'state_encoded': [DEFAULT_STATE],
                'type_encoded': [DEFAULT_TYPE],
                'day_of_week': [day_of_week_feature],
                'month': [month_feature],
                'year': [year],
                'day_of_month': [day_of_month]
            })
            
            pred = model.predict(input_data)[0]
            
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