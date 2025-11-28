# 🛒 Real-Time Retail Forecasting with RAG-Powered AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-forecast-redis.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-purple)
![Groq](https://img.shields.io/badge/AI-Groq_Llama-blue)

**[Live Demo](https://retail-forecast-redis.streamlit.app/)** | **[Architecture Diagram](retail_architecture.xml)**

---

## 📖 Overview

A production-grade MLOps system combining traditional ML with RAG for retail demand forecasting. Features real-time data streaming, automated model retraining, and AI-powered data analysis over 3M+ records.

**Key Capabilities:**

- 🔄 Live data ingestion (10-min intervals)
- 🤖 RAG-powered Q&A over 3M+ sales records
- 📊 Dual forecasting (XGBoost + Prophet)
- 🎛️ What-if scenario analysis
- 🎨 Premium glassmorphism UI
- ⚡ Zero-cost serverless infrastructure

---

## 🏗️ Architecture

![Architecture](retail_architecture.png)

---

## 🔄 How Everything Works

### **1. Data Ingestion (Every 10 Minutes)**

```
Kaggle (train.csv) → GitHub Action → producer_batch.py → Redis Stream → feature_store_batch.py → Upstash Redis
```

- Downloads 3M+ records from Kaggle
- Simulates 50 random transactions with current timestamps
- Pushes to Redis Stream
- Aggregates into daily/weekly/monthly features
- Stores in Redis for dashboard

### **2. Model Training (Nightly)**

```
Historical Data + Redis Buffer → train.py → XGBoost + Prophet → MLflow → Save Models → Git Commit → Auto-Deploy
```

- Merges Kaggle data with live Redis buffer
- Trains XGBoost on 12 features (oil, transactions, store metadata, holidays)
- Trains Prophet for long-term trends
- Saves `best_model_v2.json`, `long_term_forecast.pkl`, encoders
- Commits to repo → Streamlit Cloud auto-deploys

### **3. Dashboard Predictions**

```
User Input → Load Models → Encode Features → Fetch Redis Data → XGBoost.predict() → Display Chart
```

- User selects store/product/date
- Loads XGBoost model and encoders
- Fetches live oil price and transactions from Redis
- Runs prediction
- Shows 7-day forecast

### **4. What-If Analysis**

```
User Adjusts (Oil/Promo/Holiday) → Modify Features → XGBoost.predict() → Compare Baseline vs Scenario → Show Impact
```

- User tweaks scenario parameters
- Creates two feature sets (baseline vs scenario)
- Runs predictions for both
- Displays side-by-side comparison

### **5. Vector DB Build (Automated)**

```
train.csv → Load 500K Recent Records → Embeddings → Pinecone (Cloud) → Daily Updates
```

- Loads 500K most recent records
- Generates text: "Date: 2017-12-25, Store: 5, Product: GROCERY, Sales: $1234"
- Creates 384-dim embeddings (Sentence Transformers)
- Uploads to Pinecone via API
- Daily workflow adds new records automatically

### **6. AI Data Analyst (RAG)**

```
Question → Parse Filters → Generate Embedding → Pinecone Search → Retrieve Top-20 → Groq API → Answer
```

- User asks: "What were GROCERY sales in store 25?"
- Extracts filters: `{store_nbr: 25, family: GROCERY}`
- Searches 500K+ vectors using semantic similarity
- Retrieves top 20 matching records from Pinecone
- Sends to Groq (Llama 3.3 70B) with context
- Generates answer with citations

### **7. App Loading (First Run)**

```
User Visits → Connect Pinecone → Load Models → Connect Redis → Ready!
```

- Connects to Pinecone (cloud-hosted)
- No download needed (instant access)
- Loads ML models from repo
- Connects to Redis for live data
- App ready to serve in seconds

---

## 🛠️ Tech Stack

| Category  | Technologies                                          |
| --------- | ----------------------------------------------------- |
| **Data**  | Kaggle API, Redis Streams, Upstash Redis              |
| **ML**    | XGBoost, Prophet, Sentence Transformers               |
| **AI**    | Groq (Llama 3.3 70B), Pinecone, Sentence Transformers |
| **MLOps** | GitHub Actions, MLflow, Streamlit Cloud               |

---

## 🌟 Features

### 1. Real-Time Dashboard

- Live sales metrics from Redis
- 7-day XGBoost + 30-day Prophet forecasts
- Interactive Plotly charts

### 2. What-If Analysis

- Simulate oil price changes ($40-$120)
- Toggle promotions and holidays
- Instant prediction updates

### 3. RAG-Powered AI Analyst

- Natural language queries over 500K+ vectors
- Cloud-hosted semantic search via Pinecone
- Sub-2s responses via Groq API

**Example Questions:**

```
"What were total GROCERY sales in store 25?"
"Show sales trends for December 2017"
"Which stores had highest sales last week?"
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/RobinMillford/retail-forecast.git
cd retail-forecast
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
# Required
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token
GROQ_API_KEY=your_groq_key

# For Vector DB (Pinecone)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=retail-sales

# Optional
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 3. Run

```bash
streamlit run dashboard.py
```

### 4. Upload Data to Pinecone (Optional)

```bash
python scripts/pinecone_initial_load.py
```

---

## 📂 Project Structure

```
retail_mlops/
├── .github/workflows/       # 3 automated pipelines
├── pages/                   # What-If + AI Analyst
├── scripts/                 # Vector DB builders
├── utils/                   # Shared modules
├── dashboard.py             # Main app
├── train.py                 # Model training
└── *.joblib, *.json, *.pkl  # Model artifacts
```

---

## 🔧 API Setup

### Groq (Free)

1. Get key: https://console.groq.com/
2. Add to `.env`: `GROQ_API_KEY=gsk_...`

### Pinecone

1. Sign up: https://www.pinecone.io/
2. Create index: `retail-sales` (384 dimensions, cosine)
3. Get API key from dashboard
4. Add to `.env`:
   ```
   PINECONE_API_KEY=your-key
   PINECONE_ENVIRONMENT=us-east-1-aws
   PINECONE_INDEX_NAME=retail-sales
   ```

---

## 🎯 Performance

- **Vector DB:** 500K+ vectors, 384-dim embeddings (Pinecone)
- **Query Latency:** <2s (search + LLM)
- **Model Accuracy:** RMSE ~500
- **Uptime:** 99.9% (Streamlit Cloud + Pinecone)

---

## 🔮 Roadmap

- [ ] FastAPI deployment
- [ ] LSTM/Transformer models
- [ ] Real-time alerts
- [ ] A/B testing framework

---

## 👤 Author

**Yamin Hossain** | [GitHub](https://github.com/RobinMillford)

---

## 🙏 Credits

Kaggle • Groq • Pinecone • Streamlit • Upstash

**⭐ Star this repo if you find it helpful!**
