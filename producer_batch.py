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

# 2. Load FULL Data (Unlock the entire dataset)
print("📂 Loading full dataset for simulation...")
try:
    # FIX 1: Removed 'nrows' limit. Reads all 3M+ rows (approx 125MB)
    # FIX 2: Added 'low_memory=False' to speed up loading and fix type warnings
    df = pd.read_csv("data/train.csv", encoding='latin1', low_memory=False)
    
    # Optimization: Only keep columns we need to save RAM
    df = df[['store_nbr', 'family', 'sales', 'onpromotion']]
    
    # Get unique families for fallback if needed, but we primarily sample rows now
    print(f"  ✅ Loaded {len(df):,} sales records from 54 stores.")

except Exception as e:
    print(f"❌ CRITICAL: Failed to load data: {e}")
    exit()

print(f"🚀 Starting Batch Producer... Sending {BATCH_SIZE} random sales.")

# 3. The Batch Loop
for i in range(BATCH_SIZE):
    # FIX 3: Randomly sample 1 real transaction from the huge dataset
    # This gives us realistic correlations (e.g., Bread sells more than Automotive)
    # but allows us to run forever without "running out" of data.
    row = df.sample(1).iloc[0]
    
    # FIX 4: SYNTHETIC DATE (Time Travel)
    # We ignore the 2017 date and use "Right Now" to make the dashboard look live.
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    
    event = {
        "date": current_time,
        "store_nbr": int(row['store_nbr']),
        "family": row['family'],
        "sales": float(row['sales']),
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