from upstash_redis import Redis
import os
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
STREAM_KEY = "store_sales_stream"
GROUP_NAME = "ml_feature_group"
CONSUMER_NAME = "feature_processor_batch_1"

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
        # --- THE FIX IS HERE ---
        # This command does NOT have 'BLOCK'. It reads new messages and exits if none.
        response = redis.execute([
            "XREADGROUP", "GROUP", GROUP_NAME, CONSUMER_NAME, 
            "COUNT", "100",
            "STREAMS", STREAM_KEY, ">"  # '>' means read new messages
        ])
        
        if not response:
            # This is the expected behavior: no new data.
            print("No new data to process. Shutting down.")
            break 

        stream_data = response[0][1] 

        for message in stream_data:
            msg_id = message[0]
            fields_raw = message[1]
            
            data_dict = {fields_raw[i]: fields_raw[i+1] for i in range(0, len(fields_raw), 2)}
            
            item_family = data_dict.get('family')
            sales = float(data_dict.get('sales', 0.0))
            
            if not item_family:
                redis.execute(["XACK", STREAM_KEY, GROUP_NAME, msg_id]) 
                continue 
                
            feature_key = f"feature:sales_volume:{item_family}"
            
            redis.incrbyfloat(feature_key, sales)
            redis.execute(["XACK", STREAM_KEY, GROUP_NAME, msg_id])
            processed_count += 1
            print(f"  🔄 Processed: {item_family} | Added {sales}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        break 

print(f"\nBatch complete. Processed {processed_count} messages.")
