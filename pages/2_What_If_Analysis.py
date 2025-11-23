import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from utils import ui

# --- UI SETUP ---
ui.setup_page(page_title="What-If Analysis", page_icon="🧪")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("best_model_v2.json")
    return model

try:
    model_xgb = load_model()
    encoders = {
        "family": joblib.load("family_encoder.joblib"),
        "city": joblib.load("city_encoder.joblib"),
        "state": joblib.load("state_encoder.joblib"),
        "type": joblib.load("type_encoder.joblib")
    }
except Exception as e:
    st.error(f"Model assets missing. Please run training first.\n{e}")
    st.stop()

# --- HEADER ---
col_header_1, col_header_2 = st.columns([0.8, 0.2])
with col_header_1:
    st.title("🎛️ What-If Scenario Simulator")
    st.markdown("*Simulate business scenarios: Oil Prices, Promotions, and Holiday Events*")
with col_header_2:
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    st.button("🔄 Reset Simulation", use_container_width=True)

st.divider()

# --- MAIN LAYOUT ---
# Matching dashboard.py column ratio [0.4, 0.6]
col_controls, col_viz = st.columns([0.4, 0.6], gap="large")

with col_controls:
    st.subheader("⚙️ Scenario Parameters")
    with st.container():
        # Target Selection
        c_fam, c_store = st.columns(2)
        family = c_fam.selectbox("Item Family", encoders['family'].classes_)
        store_id = c_store.number_input("Store ID", min_value=1, max_value=54, value=1)
        
        st.markdown("---")
        
        # Variables
        oil_price = st.slider("🛢️ Oil Price ($)", 20.0, 120.0, 45.0)
        transactions = st.slider("💳 Daily Transactions", 500, 5000, 1500)
        
        c_promo, c_holiday = st.columns(2)
        is_promo = c_promo.toggle("🔥 Active Promotion", value=False)
        is_holiday = c_holiday.toggle("🎉 Holiday Event", value=False)
        
        st.markdown("---")
        
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            run_sim = True
        else:
            run_sim = False

with col_viz:
    st.subheader("📊 Impact Analysis")
    if run_sim:
        # Prepare Data
        dates = pd.date_range(start=datetime.now(), periods=7)
        
        # Mock Store Metadata (Simplified)
        city_enc = 0 
        state_enc = 0
        type_enc = 0
        
        fam_enc = encoders['family'].transform([family])[0]
        
        preds = []
        baseline_preds = []
        
        for d in dates:
            # Scenario Input
            row = pd.DataFrame([{
                'store_nbr': store_id, 'family_encoded': fam_enc,
                'onpromotion': 1 if is_promo else 0, 'transactions': transactions,
                'dcoilwtico': oil_price, 'is_holiday': 1 if is_holiday else 0,
                'city_encoded': city_enc, 'state_encoded': state_enc,
                'type_encoded': type_enc, 'day_of_week': d.weekday(),
                'month': d.month, 'year': d.year, 'day_of_month': d.day
            }])
            
            # Baseline Input (Standard values)
            row_base = row.copy()
            row_base['dcoilwtico'] = 45.0 # Average
            row_base['transactions'] = 1500 # Average
            row_base['onpromotion'] = 0
            
            p_scenario = max(0, model_xgb.predict(row)[0])
            p_base = max(0, model_xgb.predict(row_base)[0])
            
            preds.append(p_scenario)
            baseline_preds.append(p_base)
            
        # Calculate Impact
        total_base = sum(baseline_preds)
        total_scen = sum(preds)
        diff = total_scen - total_base
        pct = (diff / total_base) * 100 if total_base > 0 else 0
        
        # --- METRICS ROW ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Sales", f"${total_base:,.0f}", "7 Days")
        m2.metric("Scenario Sales", f"${total_scen:,.0f}", f"{pct:+.1f}%", delta_color="normal" if diff > 0 else "inverse")
        m3.metric("Net Impact", f"${diff:,.0f}", "Revenue Gain" if diff > 0 else "Revenue Loss")
        
        st.markdown("---")

        # --- PLOT ---
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=dates, y=baseline_preds,
            name='Baseline',
            marker_color='#30363D'
        ))
        
        fig.add_trace(go.Bar(
            x=dates, y=preds,
            name='Scenario',
            marker_color='#238636'
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode='group',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Empty State
        with st.container():
            st.info("👈 Adjust parameters on the left and click 'Run Simulation' to see the impact.")
            # Using a placeholder image or just text to keep it clean
            st.caption("Select a product family and store to begin.")
