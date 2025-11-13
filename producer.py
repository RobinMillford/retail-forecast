import pandas as pd
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import time
import upstash_redis

# 1. Load Environment Variables
load_dotenv()

print(f"ℹ️ Using upstash-redis version: {upstash_redis.__version__}")

url = os.getenv("UPSTASH_REDIS_REST_URL")
token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

# 2. Connect to Redis
print("🔌 Connecting to Redis...")
redis = Redis(url=url, token=token)

# 3. Prepare the "Future" Data
print("📂 Loading and filtering data...")
try:
    df = pd.read_csv("data/train.csv")
except FileNotFoundError:
    print("❌ Error: 'data/train.csv' not found. Make sure you are in the retail_mlops folder.")
    exit()

df['date'] = pd.to_datetime(df['date'])

# Filter for 2017 data (The "Future") and Store #1
stream_df = df[(df['date'].dt.year == 2017) & (df['store_nbr'] == 1)].sort_values('date')

print(f"🚀 Starting Stream! Simulating {len(stream_df)} transactions...")
print("Press Ctrl+C to stop.")

# 4. The Streaming Loop
for index, row in stream_df.iterrows():
    event = {
        "date": str(row['date'].date()),
        "store_nbr": int(row['store_nbr']),
        "family": row['family'],
        "sales": float(row['sales']),
        "onpromotion": int(row['onpromotion'])
    }
    
    try:
        # --- THE FIX: RAW COMMAND MODE ---
        # We convert the dictionary to a flat list: ['date', '2017-01-01', 'sales', '10.0', ...]
        # This bypasses the library version conflict by sending the raw Redis command.
        
        flat_args = []
        for k, v in event.items():
            flat_args.extend([str(k), str(v)])
            
        # Command: XADD stream_name * field value field value ...
        redis.execute(["XADD", "store_sales_stream", "*"] + flat_args)
        
        print(f"✅ Sent: {event['date']} | Item: {event['family'][:10]}.. | Sales: {event['sales']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        # If it fails, print the full error to help debug
        import traceback
        traceback.print_exc()
        break

    # Speed up the demo: 0.05s sleep = 20 items per second
    time.sleep(0.05)