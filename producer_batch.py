import pandas as pd
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import random
import datetime

# --- CONFIG ---
load_dotenv()
STREAM_KEY = "store_sales_stream"
BATCH_SIZE = 50 

# 1. Connect to Redis
print("🔌 Connecting to Redis...")
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    redis.ping()
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")
    exit()

# 2. Load SIMULATION Data (from test.csv to prevent data duplication)
print("📂 Loading 'data/test.csv' for simulation...")
try:
    # --- THIS IS THE FIX ---
    # We now read from test.csv. This file has NO sales data,
    # so we are truly simulating new, unseen events.
    df = pd.read_csv("data/test.csv", encoding='latin1', low_memory=False)
    
    # Optimization: Only keep columns we need
    df = df[['store_nbr', 'family', 'onpromotion']]
    
    print(f"  ✅ Loaded {len(df):,} future events to simulate.")

except Exception as e:
    print(f"❌ CRITICAL: Failed to load data: {e}")
    exit()

print(f"🚀 Starting Batch Producer... Sending {BATCH_SIZE} simulated sales.")

# 3. The Batch Loop
for i in range(BATCH_SIZE):
    # Randomly sample 1 event from the "future" (test.csv)
    row = df.sample(1).iloc[0]
    
    # --- SYNTHETIC DATA GENERATION ---
    # 1. Use today's date to make it "live"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 2. Invent a realistic sales number
    # We invent a base sale. If it's on promo, we boost it.
    base_sales = random.uniform(5.0, 500.0)
    if row['onpromotion'] == 1:
        base_sales *= 1.5 # 50% sales boost for promo
        
    simulated_sales = round(base_sales, 2)
    # ---------------------------------
    
    event = {
        "date": current_time,
        "store_nbr": int(row['store_nbr']),
        "family": row['family'],
        "sales": simulated_sales, # Use the new simulated sales
        "onpromotion": int(row['onpromotion']),
        "batch_id": i
    }
    
    flat_args = []
    for k, v in event.items():
        flat_args.extend([str(k), str(v)])
        
    try:
        redis.execute(["XADD", STREAM_KEY, "*"] + flat_args)
        # Short log to save space
        print(f"  ✅ {event['date']} | Store {event['store_nbr']} | {event['family'][:10]}.. | ${event['sales']}")
    except Exception as e:
        print(f"  ❌ Error sending to stream: {e}")

print(f"\nBatch complete. Sent {BATCH_SIZE} events. Producer shutting down.")