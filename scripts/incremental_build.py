"""
Incremental Vector Database Builder
Builds ChromaDB in chunks and uploads to Hugging Face Hub progressively
"""

import pandas as pd
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector_db import init_chroma_db, add_records_to_db
from utils.hf_storage import HFVectorStorage

# Configuration
BATCH_SIZE = 300000  # Process 300K records at a time
PROGRESS_FILE = "build_progress.json"

def load_progress():
    """Load build progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"last_processed_index": 0, "total_processed": 0, "batches_completed": 0}

def save_progress(progress):
    """Save build progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def main():
    print("=" * 60)
    print("🔄 Incremental Vector Database Builder")
    print("=" * 60)
    
    # Load progress
    progress = load_progress()
    print(f"\n📊 Current Progress:")
    print(f"  - Last processed index: {progress['last_processed_index']:,}")
    print(f"  - Total processed: {progress['total_processed']:,}")
    print(f"  - Batches completed: {progress['batches_completed']}")
    
    # 1. Load historical data
    print("\n📂 Loading historical data...")
    try:
        df_history = pd.read_csv("data/train.csv", encoding='latin1', low_memory=False)
        df_history['date'] = pd.to_datetime(df_history['date'])
        
        # Load metadata
        df_stores = pd.read_csv("data/stores.csv")
        df_oil = pd.read_csv("data/oil.csv")
        df_oil['date'] = pd.to_datetime(df_oil['date'])
        df_oil = df_oil.set_index('date').resample('D').ffill().reset_index()
        
        df_holidays = pd.read_csv("data/holidays_events.csv")
        df_holidays['date'] = pd.to_datetime(df_holidays['date'])
        df_holidays = df_holidays[df_holidays['transferred'] == False]
        df_holidays['is_holiday'] = 1
        
        print(f"  ✅ Loaded {len(df_history):,} historical records")
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return
    
    # 2. Merge with metadata
    print("\n🔗 Merging with metadata...")
    df = pd.merge(df_history, df_stores, on='store_nbr', how='left')
    df = pd.merge(df, df_oil, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
    df = pd.merge(df, df_holidays[['date', 'is_holiday']], on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    df = df.sort_values('date').reset_index(drop=True)
    
    total_records = len(df)
    print(f"  ✅ Total records: {total_records:,}")
    
    # 3. Check if already complete
    if progress['last_processed_index'] >= total_records:
        print("\n✅ All records already processed!")
        print("   Run with --reset to rebuild from scratch")
        return
    
    # 4. Download existing DB from Hugging Face (if exists)
    hf_repo_id = os.getenv("HF_REPO_ID")
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_repo_id:
        print("\n⚠️ HF_REPO_ID not set. Set it in .env or environment variables.")
        print("   Example: HF_REPO_ID=username/retail-sales-vector-db")
        return
    
    storage = HFVectorStorage(repo_id=hf_repo_id, token=hf_token)
    
    # Try to download existing DB
    if progress['batches_completed'] > 0:
        print(f"\n📥 Downloading existing database from Hugging Face...")
        try:
            storage.download_vector_db(local_db_path="./chroma_db", force=True)
        except Exception as e:
            print(f"  ⚠️ Could not download existing DB: {e}")
            print("  Starting fresh...")
    
    # 5. Initialize ChromaDB
    print("\n🗄️ Initializing ChromaDB...")
    collection = init_chroma_db(persist_directory="./chroma_db")
    existing_count = collection.count()
    print(f"  Current records in DB: {existing_count:,}")
    
    # 6. Process next batch
    start_idx = progress['last_processed_index']
    end_idx = min(start_idx + BATCH_SIZE, total_records)
    batch_df = df.iloc[start_idx:end_idx]
    
    print(f"\n🧮 Processing batch {progress['batches_completed'] + 1}:")
    print(f"  Records: {start_idx:,} to {end_idx:,} ({len(batch_df):,} records)")
    
    start_time = datetime.now()
    add_records_to_db(collection, batch_df, batch_size=5000)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"  ⏱️ Batch completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # 7. Update progress
    progress['last_processed_index'] = end_idx
    progress['total_processed'] = collection.count()
    progress['batches_completed'] += 1
    save_progress(progress)
    
    # 8. Upload to Hugging Face
    print(f"\n📤 Uploading to Hugging Face Hub...")
    commit_msg = f"Batch {progress['batches_completed']}: {progress['total_processed']:,}/{total_records:,} records ({progress['total_processed']/total_records*100:.1f}%)"
    storage.upload_vector_db(local_db_path="./chroma_db", commit_message=commit_msg)
    
    # 9. Summary
    print("\n" + "=" * 60)
    print("✅ Batch Complete!")
    print("=" * 60)
    print(f"  📊 Progress: {progress['total_processed']:,} / {total_records:,} ({progress['total_processed']/total_records*100:.1f}%)")
    print(f"  🔢 Batches completed: {progress['batches_completed']}")
    
    remaining = total_records - progress['total_processed']
    if remaining > 0:
        est_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  ⏳ Remaining: {remaining:,} records (~{est_batches} more batches)")
    else:
        print(f"  🎉 ALL HISTORICAL RECORDS PROCESSED!")
        
        # --- ADD LIVE DATA FROM REDIS ---
        print("\n" + "=" * 60)
        print("📡 Fetching Live Data from Redis...")
        print("=" * 60)
        
        try:
            from upstash_redis import Redis
            
            # Connect to Redis
            redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
            redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
            
            if not redis_url or not redis_token:
                print("  ⚠️ Redis credentials not found. Skipping live data.")
            else:
                redis_client = Redis(url=redis_url, token=redis_token)
                
                # Fetch training buffer (live data)
                buffer_data = redis_client.get("training_buffer")
                
                if buffer_data:
                    import json
                    live_records = json.loads(buffer_data)
                    
                    if live_records and len(live_records) > 0:
                        print(f"  ✅ Found {len(live_records)} live records in Redis")
                        
                        # Convert to DataFrame
                        live_df = pd.DataFrame(live_records)
                        
                        # Ensure required columns
                        if 'date' in live_df.columns and 'store_nbr' in live_df.columns:
                            # Add to vector DB
                            print(f"\n📥 Adding {len(live_df)} live records to vector database...")
                            add_records_to_db(collection, live_df)
                            
                            # Update progress
                            progress['total_processed'] = collection.count()
                            progress['live_data_added'] = len(live_df)
                            save_progress(progress)
                            
                            # Upload to Hugging Face
                            print(f"\n📤 Uploading updated database with live data...")
                            commit_msg = f"Complete: {progress['total_processed']:,} records (historical + {len(live_df)} live)"
                            storage.upload_vector_db(local_db_path="./chroma_db", commit_message=commit_msg)
                            
                            print(f"\n  ✅ Live data successfully integrated!")
                            print(f"  📊 Total records: {progress['total_processed']:,}")
                        else:
                            print("  ⚠️ Live data missing required columns (date, store_nbr)")
                    else:
                        print("  ℹ️ No live data found in Redis buffer")
                else:
                    print("  ℹ️ No training buffer found in Redis")
        
        except Exception as e:
            print(f"  ❌ Error fetching live data: {e}")
            print("  ℹ️ Continuing without live data...")


if __name__ == "__main__":
    main()
