# 🛒 Real-Time Retail Forecasting with RAG-Powered AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-forecast-redis.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
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

### **5. Vector DB Build (Incremental)**

```
train.csv → Batch 1 (300K) → Embeddings → ChromaDB → Upload to HF → Batch 2 → ... → Complete (3M+)
```

- Processes 300K records at a time
- Generates text: "Date: 2017-12-25, Store: 5, Product: GROCERY, Sales: $1234"
- Creates 384-dim embeddings (Sentence Transformers)
- Stores in ChromaDB with metadata
- Uploads batch to Hugging Face Hub
- Repeats until all 3M+ records done

### **6. AI Data Analyst (RAG)**

```
Question → Parse Filters → Generate Embedding → ChromaDB Search → Retrieve Top-20 → Groq API → Answer
```

- User asks: "What were GROCERY sales in store 25?"
- Extracts filters: `{store_nbr: 25, family: GROCERY}`
- Searches 3M+ records using semantic similarity
- Retrieves top 20 matching records
- Sends to Groq (Llama 3.3 70B) with context
- Generates answer with citations

### **7. App Loading (First Run)**

```
User Visits → Check ChromaDB → Download from HF (if missing) → Cache → Load Models → Connect Redis → Ready!
```

- Checks for local ChromaDB
- Downloads from Hugging Face (one-time, ~5 min)
- Streamlit Cloud caches database
- Loads ML models from repo
- Connects to Redis
- App ready to serve

---

## 🛠️ Tech Stack

| Category  | Technologies                                     |
| --------- | ------------------------------------------------ |
| **Data**  | Kaggle API, Redis Streams, Upstash Redis         |
| **ML**    | XGBoost, Prophet, Sentence Transformers          |
| **AI**    | Groq (Llama 3.3 70B), ChromaDB, Hugging Face Hub |
| **MLOps** | GitHub Actions, MLflow, Streamlit Cloud          |

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

- Natural language queries over 3M+ records
- Semantic search with metadata filtering
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

# For Vector DB
HF_REPO_ID=username/retail-sales-vector-db
HF_TOKEN=your_hf_token

# Optional
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 3. Run

```bash
streamlit run dashboard.py
```

### 4. Build Vector DB (Optional)

```bash
python scripts/incremental_build.py
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

### Hugging Face

1. Create token: https://huggingface.co/settings/tokens
2. Create dataset: https://huggingface.co/new-dataset
3. Add to `.env`:
   ```
   HF_REPO_ID=username/retail-sales-vector-db
   HF_TOKEN=hf_...
   ```

---

## 🎯 Performance

- **Vector DB:** 3M+ records, 384-dim embeddings
- **Query Latency:** <2s (search + LLM)
- **Model Accuracy:** RMSE ~500
- **Uptime:** 99.9% (Streamlit Cloud)

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

Kaggle • Groq • Hugging Face • Streamlit • Upstash

**⭐ Star this repo if you find it helpful!**
