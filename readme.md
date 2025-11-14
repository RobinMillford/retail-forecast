# 🛒 End-to-End Real-Time Retail Demand Forecasting System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-forecast-redis.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Prophet](https://img.shields.io/badge/ML-Prophet-orange)
![Redis](https://img.shields.io/badge/Database-Upstash_Redis-red)
![MLOps](https://img.shields.io/badge/MLOps-GitHub_Actions-purple)

### 🚀 Live Demo

**[Click here to view the Live Dashboard](https://retail-forecast-redis.streamlit.app/)**
_(Note: If the live data is $0, hit the "Refresh" button. The background pipeline updates every 5 minutes.)_

---

## 📖 Project Overview

This project is a production-grade **MLOps pipeline** designed to forecast retail sales in near-real-time. Unlike static notebooks, this system is a living ecosystem that ingests live data, updates a feature store, automatically retrains multiple models nightly on fresh data, and serves predictions via an interactive dashboard.

**The Business Problem:** Retailers struggle with inventory management because demand fluctuates rapidly due to promotions, holidays, and oil prices.
**The Solution:** A self-healing AI system that adapts to new data trends automatically using a **Hybrid Forecasting Strategy** (XGBoost + Prophet).

---

## 🏗️ System Architecture

I designed a **Serverless Micro-Batch Architecture** to maintain a production-grade pipeline with **$0 operational costs**.

![Retail Architecture](retail_architecture.png)

---

## 🛠️ Tech Stack

- **Data Ingestion:** Kaggle API (Automated ETL), Redis Streams (Message Queue).
- **Feature Store:** Upstash Redis (Serverless Key-Value Store).
- **Machine Learning:** \* **XGBoost:** For high-precision, feature-rich daily predictions.
  - **Prophet:** For long-term trend and seasonality forecasting.
- **Experiment Tracking:** MLflow & Dagshub.
- **Orchestration (CI/CD):** GitHub Actions (Scheduled Cron Jobs).
- **Frontend:** Streamlit (Python-based UI) with Plotly charts.
- **Language:** Python 3.11.

---

## ⚙️ Key Features & Engineering Decisions

### 1. Serverless "Micro-Batch" Ingestion

Instead of running an expensive 24/7 EC2 instance, I utilized **GitHub Actions** as a cron scheduler.

- **Producer:** Wakes up every 5 minutes, downloads the latest dataset, simulates 50 random transactions with _current timestamps_ (Time-Travel Logic), and pushes them to Redis.
- **Processor:** Consumes the stream, updates the Feature Store, and buffers raw data for nightly training.

### 2. The "Data Flywheel" (Continuous Training)

Most MLOps projects only train on static history. This project implements a true **Data Flywheel**:

1.  **Day:** Live data flows into a Redis List (`training_buffer`).
2.  **Night:** The training script downloads the 3M row historical dataset from Kaggle.
3.  **Merge:** It pulls the fresh rows from Redis, combines them with history, and trains the models on the **combined dataset**.
4.  **Deploy:** The new models are committed to the repo, keeping the AI up-to-date with recent trends.

### 3. Hybrid Model Strategy

One model cannot do it all. I implemented a dual-model approach:

- **XGBoost (v2):** Trained on 12 complex features (Oil Price, Transactions, Store Metadata, Holidays). Used for **Single-Day Precision**.
- **Prophet:** Trained on aggregated time-series data. Used for **30-Day Trend Forecasting** and uncertainty visualization.

### 4. Time-Bucketed Feature Store

The processor aggregates raw sales into three distinct time buckets in Redis for low-latency dashboard queries:

- `feature:sales_daily`
- `feature:sales_weekly`
- `feature:sales_monthly`

---

## 📂 Project Structure

```bash
├── .github/workflows/
│   ├── live_stream.yml    # Runs every 5 mins (Ingestion & Feature Store)
│   └── retrain.yml        # Runs daily (CT: Trains XGBoost & Prophet)
├── data/                  # (Ignored by Git) Holds temp Kaggle data
├── dashboard.py           # Streamlit Frontend (Visualizes Live Data + Forecasts)
├── producer_batch.py      # Generates synthetic live data using test.csv
├── feature_store_batch.py # Aggregates data & saves to Training Buffer
├── train.py               # Advanced training logic (Flywheel + Multi-Model)
├── best_model_v2.json     # The production XGBoost artifact
├── long_term_forecast.pkl # The production Prophet artifact
├── *_encoder.joblib       # Saved LabelEncoders for inference
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🏃‍♂️ How to Run Locally

If you want to run this pipeline on your own machine:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/RobinMillford/retail-forecast.git
    cd retail-forecast
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory with the following keys:

    ```ini
    UPSTASH_REDIS_REST_URL="your_url_here"
    UPSTASH_REDIS_REST_TOKEN="your_token_here"
    MLFLOW_TRACKING_URI="your_dagshub_uri"
    MLFLOW_TRACKING_USERNAME="your_username"
    MLFLOW_TRACKING_PASSWORD="your_token"
    KAGGLE_USERNAME="your_kaggle_user"
    KAGGLE_KEY="your_kaggle_key"
    ```

4.  **Run the Dashboard:**

    ```bash
    streamlit run dashboard.py
    ```

5.  **Simulate Data (Optional):**
    Open a separate terminal and run:

    ```bash
    python producer_batch.py
    python feature_store_batch.py
    ```

---

## 🔮 Future Improvements

- **Drift Monitoring:** Integrate **Evidently AI** to send alerts when the data distribution shifts significantly.
- **API Deployment:** Wrap the XGBoost model in a **FastAPI** container for serving predictions via REST.
- **Deep Learning:** Experiment with LSTM Transformers for sequence modeling on the largest stores.

---

_Built by Yamin_
