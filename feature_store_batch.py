from upstash_redis import Redis
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import json

# --- CONFIG ---
load_dotenv()
STREAM_KEY = "store_sales_stream"
GROUP_NAME = "ml_feature_group"
CONSUMER_NAME = "feature_processor_batch_1"
TRAINING_BUFFER_KEY = "training_data_buffer" # <-- NEW KEY FOR CONTINUOUS TRAINING

# 1. Connect to Redis
print("🔌 Connecting to Redis...")
try:
    redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))
    redis.ping()
except Exception as e:
    print(f"❌ Failed to connect to Redis: {e}")
    exit()

# 2. Create Consumer Group
print(f"ℹ️ Ensuring Consumer Group '{GROUP_NAME}' exists...")
try:
    redis.execute(["XGROUP", "CREATE", STREAM_KEY, GROUP_NAME, "0", "MKSTREAM"])
    print("  ✅ Consumer Group Created!")
except Exception as e:
    if "BUSYGROUP" in str(e):
        print("  ℹ️ Consumer Group already exists.")
    else:
        print(f"  ❌ Error creating group: {e}")
        exit()

print(f"🚀 Starting Batch Processor for consumer '{CONSUMER_NAME}'...")

# 3. Batch Processing Loop
processed_count = 0
while True:
    try:
        response = redis.execute([
            "XREADGROUP", "GROUP", GROUP_NAME, CONSUMER_NAME, 
            "COUNT", "100",
            "STREAMS", STREAM_KEY, ">" 
        ])
        
        if not response:
            print("No new data to process. Shutting down.")
            break 

        stream_data = response[0][1] 

        for message in stream_data:
            msg_id = message[0]
            fields_raw = message[1]
            data_dict = {fields_raw[i]: fields_raw[i+1] for i in range(0, len(fields_raw), 2)}
            
            item_family = data_dict.get('family')
            store_nbr = data_dict.get('store_nbr', '?')
            event_date_str = data_dict.get('date', None)
            sales = float(data_dict.get('sales', 0.0))
            
            if not item_family or not event_date_str:
                redis.execute(["XACK", STREAM_KEY, GROUP_NAME, msg_id]) 
                continue 
            
            # --- TASK 1: UPDATE DASHBOARD AGGREGATES ---
            try:
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
                key_daily = f"feature:sales_daily:{item_family}:{event_date.strftime('%Y-%m-%d')}"
                key_weekly = f"feature:sales_weekly:{item_family}:{event_date.strftime('%Y-W%U')}"
                key_monthly = f"feature:sales_monthly:{item_family}:{event_date.strftime('%Y-%m')}"
                
                redis.incrbyfloat(key_daily, sales)
                redis.incrbyfloat(key_weekly, sales)
                redis.incrbyfloat(key_monthly, sales)

            except ValueError:
                print(f"  SKIPPED: Invalid date format {event_date_str}")
                redis.execute(["XACK", STREAM_KEY, GROUP_NAME, msg_id]) 
                continue

            # --- TASK 2 (NEW): SAVE RAW DATA FOR TRAINING ---
            # We save the original dictionary (as a JSON string) to our training buffer list
            redis.lpush(TRAINING_BUFFER_KEY, json.dumps(data_dict))
            
            # --- TASK 3: ACKNOWLEDGE MESSAGE ---
            redis.execute(["XACK", STREAM_KEY, GROUP_NAME, msg_id])
            processed_count += 1
            
            print(f"  🔄 Processed & Saved: {event_date_str} | Store {store_nbr} | {item_family} | +${sales}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        break 

# After processing, we'll trim the buffer to ~100k records to save memory
# This keeps the last ~2 days of data for training (100 * 288 runs/day)
redis.ltrim(TRAINING_BUFFER_KEY, 0, 100000)
print(f"\nBatch complete. Processed {processed_count} messages. Training buffer trimmed.")