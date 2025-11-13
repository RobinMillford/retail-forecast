import pandas as pd
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import random

# --- CONFIG ---
load_dotenv()
STREAM_KEY = "store_sales_stream"
BATCH_SIZE = 50 # Send 50 sales every time this script runs

# 1. Connect to Redis
print("🔌 Connecting to Redis...")
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    redis.ping()
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")
    exit()

# 2. Prepare "Future" Data (as a source of items)
print("📂 Loading data for simulation...")
try:
    df = pd.read_csv("data/train.csv", nrows=50000)
    df = df[(df['store_nbr'] == 1) & (df['family'] != 'OTHER')]
    # Get a list of unique items to simulate
    item_families = df['family'].unique()
    if len(item_families) == 0:
        raise Exception("No item families found in data.")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit()

print(f"🚀 Starting Batch Producer... Sending {BATCH_SIZE} random sales.")

# 3. The Batch Loop (sends 50 items and stops)
for i in range(BATCH_SIZE):
    item = random.choice(item_families)
    
    event = {
        "store_nbr": 1,
        "family": item,
        "sales": round(random.uniform(5.0, 500.0), 2),
        "onpromotion": random.randint(0, 1),
        "batch_id": i
    }
    
    # 4. Convert dict to flat list for raw execute
    flat_args = []
    for k, v in event.items():
        flat_args.extend([str(k), str(v)])
        
    # 5. Send to Stream
    try:
        redis.execute(["XADD", STREAM_KEY, "*"] + flat_args)
        print(f"  ✅ Sent: {event['family'][:10]}.. | Sales: {event['sales']}")
    except Exception as e:
        print(f"  ❌ Error sending to stream: {e}")

print(f"\nBatch complete. Sent {BATCH_SIZE} events. Producer shutting down.")