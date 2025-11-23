import pandas as pd
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import random
import datetime

# --- CONFIG ---
load_dotenv()
STREAM_KEY = "store_sales_stream"
DAYS_TO_BACKFILL = 7
EVENTS_PER_DAY = 100 # Generate 100 sales per day

# 1. Connect to Redis
print("🔌 Connecting to Redis...")
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    redis.ping()
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")
    exit()

# 2. Load SIMULATION Data
print("📂 Loading 'data/test.csv' for simulation...")
try:
    df = pd.read_csv("data/test.csv", encoding='latin1', low_memory=False)
    df = df[['store_nbr', 'family', 'onpromotion']]
    print(f"  ✅ Loaded {len(df):,} items to simulate.")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load data: {e}")
    exit()

print(f"🚀 Starting Backfill... Generating {EVENTS_PER_DAY} events for the last {DAYS_TO_BACKFILL} days.")

# 3. Backfill Loop
today = datetime.datetime.now()

for day_offset in range(DAYS_TO_BACKFILL):
    # Calculate date (Today, Yesterday, Day Before...)
    target_date = today - datetime.timedelta(days=day_offset)
    date_str = target_date.strftime("%Y-%m-%d")
    
    print(f"  📅 Simulating data for: {date_str}")
    
    for i in range(EVENTS_PER_DAY):
        # Randomly sample 1 event
        row = df.sample(1).iloc[0]
        
        # Invent realistic sales
        base_sales = random.uniform(5.0, 500.0)
        if row['onpromotion'] == 1:
            base_sales *= 1.5 
            
        simulated_sales = round(base_sales, 2)
        
        event = {
            "date": date_str,
            "store_nbr": int(row['store_nbr']),
            "family": row['family'],
            "sales": simulated_sales,
            "onpromotion": int(row['onpromotion']),
            "batch_id": f"backfill_{day_offset}_{i}"
        }
        
        flat_args = []
        for k, v in event.items():
            flat_args.extend([str(k), str(v)])
            
        try:
            redis.execute(["XADD", STREAM_KEY, "*"] + flat_args)
        except Exception as e:
            print(f"  ❌ Error sending to stream: {e}")

print(f"\n✅ Backfill complete! {DAYS_TO_BACKFILL * EVENTS_PER_DAY} events sent to Redis.")
print("👉 Now run 'python feature_store_batch.py' to aggregate this data into the Feature Store.")
