import streamlit as st

def setup_page(page_title="Retail AI", page_icon="🛒"):
    """
    Applies the shared Premium Modern Theme and standard page configuration.
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    # --- PREMIUM MODERN THEME CSS ---
    st.markdown(
        """
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        /* GLOBAL RESET & FONT */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* MAIN BACKGROUND - Cosmic Gradient */
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #FAFAFA;
        }
        
        /* CONTAINERS & CARDS - Glassmorphism */
        .stContainer, [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* METRIC CARDS - Premium Style */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        div[data-testid="stMetricLabel"] {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF;
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 13px;
            font-weight: 600;
        }
        
        /* HEADERS - Gradient Text */
        h1, h2, h3 {
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h1 {
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            font-size: 1.8rem;
        }
        h3 {
            font-size: 1.3rem;
        }
        
        /* BUTTONS - Gradient with Hover Effect */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            padding: 12px 28px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* INPUT FIELDS - Glassmorphism */
        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stDateInput > div > div > input {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            color: white;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* SLIDERS */
        .stSlider > div > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* TABS - Modern Style */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            color: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-color: transparent !important;
        }
        
        /* SIDEBAR - Dark Glassmorphism */
        section[data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.95);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* DATAFRAME - Modern Table */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.02);
        }
        
        /* EXPANDER - Glassmorphism */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* SCROLLBAR - Gradient */
        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        
        /* DIVIDER */
        hr {
            border-color: rgba(255, 255, 255, 0.1);
        }
        
        /* SPINNER */
        .stSpinner > div {
            border-color: #667eea !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

def format_metric(label, value, delta=None, prefix="$"):
    """Helper to display consistent metrics."""
    val_str = f"{prefix}{value:,.2f}" if isinstance(value, (int, float)) else str(value)
    st.metric(label=label, value=val_str, delta=delta)

