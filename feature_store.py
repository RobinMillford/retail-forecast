import time
from upstash_redis import Redis
import os
from dotenv import load_dotenv
import json

# 1. Connect to Redis
load_dotenv()
redis = Redis(url=os.getenv("UPSTASH_REDIS_REST_URL"), token=os.getenv("UPSTASH_REDIS_REST_TOKEN"))

STREAM_KEY = "store_sales_stream"
GROUP_NAME = "ml_feature_group"
CONSUMER_NAME = "feature_processor_1"

# 2. Create Consumer Group (Only needs to happen once)
try:
    # XGROUP CREATE stream group_name $ MKSTREAM
    # '$' means "start reading from new messages only" (or '0' for all history)
    redis.execute(["XGROUP", "CREATE", STREAM_KEY, GROUP_NAME, "0", "MKSTREAM"])
    print("✅ Consumer Group Created!")
except Exception as e:
    print("ℹ️ Consumer Group already exists (this is fine).")

print("🚀 Starting Feature Processor...")
print("I will read sales and calculate '7-Day Rolling Volume'...")

# 3. Processing Loop
while True:
    try:
        # Read new messages as part of the group
        # XREADGROUP GROUP group consumer COUNT 1 STREAMS stream >
        response = redis.execute([
            "XREADGROUP", "GROUP", GROUP_NAME, CONSUMER_NAME, 
            "COUNT", "5", "STREAMS", STREAM_KEY, ">"
        ])
        
        if not response:
            # No new data? Sleep a bit
            time.sleep(1)
            continue

        # 'response' structure is complex: [[stream_name, [[id, [fields...]], ...]]]
        stream_data = response[0][1] 

        for message in stream_data:
            msg_id = message[0]
            fields_raw = message[1]
            
            # Convert raw list [key, val, key, val] to dict
            # (Because we used raw execute, we get a raw list back)
            data_dict = {fields_raw[i]: fields_raw[i+1] for i in range(0, len(fields_raw), 2)}
            
            # --- FEATURE ENGINEERING LOGIC ---
            item_family = data_dict.get('family')
            sales = float(data_dict.get('sales'))
            
            # We want to track "Total Sales" for this Item Category
            # In a real app, we would use a Sliding Window. 
            # Here, for the MVP, we will increment a counter in Redis.
            
            feature_key = f"feature:sales_volume:{item_family}"
            
            # INCRBYFLOAT is perfect for aggregation
            redis.incrbyfloat(feature_key, sales)
            
            # Acknowledge the message (tell Redis "I processed this")
            redis.execute(["XACK", STREAM_KEY, GROUP_NAME, msg_id])
            
            print(f"🔄 Updated {item_family}: Added {sales}")

    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(1)