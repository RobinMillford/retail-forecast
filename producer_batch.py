import pandas as pd
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import random

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

# 2. Prepare Item List for Simulation
print("📂 Loading 'data/train.csv' for simulation...")
try:
    # --- THE FIX IS HERE ---
    # Added encoding='latin1' to handle special characters
    df = pd.read_csv("data/train.csv", encoding='latin1', nrows=50000) 
    
    df = df[(df['store_nbr'] == 1) & (df['family'] != 'OTHER')]
    item_families = df['family'].unique()
    if len(item_families) == 0:
        raise Exception("No item families found in data.")
    print(f"  ✅ Loaded {len(item_families)} item families.")

except Exception as e:
    print(f"❌ CRITICAL: Failed to load 'data/train.csv': {e}")
    exit()


print(f"🚀 Starting Batch Producer... Sending {BATCH_SIZE} random sales.")

# 3. The Batch Loop
for i in range(BATCH_SIZE):
    item = random.choice(item_families)
    
    event = {
        "store_nbr": 1,
        "family": item,
        "sales": round(random.uniform(5.0, 500.0), 2),
        "onpromotion": random.randint(0, 1),
        "batch_id": i
    }
    
    flat_args = []
    for k, v in event.items():
        flat_args.extend([str(k), str(v)])
        
    try:
        redis.execute(["XADD", STREAM_KEY, "*"] + flat_args)
        print(f"  ✅ Sent: {event['family'][:10]}.. | Sales: {event['sales']}")
    except Exception as e:
        print(f"  ❌ Error sending to stream: {e}")

print(f"\nBatch complete. Sent {BATCH_SIZE} events. Producer shutting down.")
