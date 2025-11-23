"""
One-time script to build the vector database from historical sales data.
Run this once to create the ChromaDB collection.

Usage:
    python scripts/build_vector_db.py
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector_db import init_chroma_db, add_records_to_db

def main():
    print("=" * 60)
    print("🚀 Building Vector Database for RAG-Powered AI Analyst")
    print("=" * 60)
    
    start_time = datetime.now()
    
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
        print("\n💡 Make sure you have run the data download first:")
        print("   - Download from Kaggle or run the GitHub Action workflow")
        return
    
    # 2. Merge with metadata
    print("\n🔗 Merging with store metadata, oil prices, and holidays...")
    df = pd.merge(df_history, df_stores, on='store_nbr', how='left')
    df = pd.merge(df, df_oil, on='date', how='left')
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
    df = pd.merge(df, df_holidays[['date', 'is_holiday']], on='date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(0)
    
    print(f"  ✅ Merged dataset ready: {len(df):,} rows")
    
    # 3. SMART SAMPLING - Reduce size while keeping quality
    print("\n🎯 Applying Smart Sampling Strategy...")
    print("  Strategy: Keep ALL recent data + sample older data")
    
    # Define cutoff: Keep all data from last 6 months
    max_date = df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=180)  # 6 months
    
    # Split data
    recent_df = df[df['date'] >= cutoff_date]
    old_df = df[df['date'] < cutoff_date]
    
    print(f"  📅 Recent data (last 6 months): {len(recent_df):,} rows - KEEPING ALL")
    print(f"  📅 Older data: {len(old_df):,} rows - SAMPLING")
    
    # Sample older data intelligently
    # Keep 20% of older data, stratified by store and family
    sample_fraction = 0.20
    old_df_sampled = old_df.groupby(['store_nbr', 'family'], group_keys=False).apply(
        lambda x: x.sample(frac=sample_fraction, random_state=42) if len(x) > 10 else x
    )
    
    print(f"  ✂️ Sampled older data: {len(old_df_sampled):,} rows ({sample_fraction*100:.0f}% of older data)")
    
    # Combine
    df_final = pd.concat([recent_df, old_df_sampled], ignore_index=True)
    df_final = df_final.sort_values('date').reset_index(drop=True)
    
    print(f"\n  ✅ Final dataset: {len(df_final):,} rows (reduced from {len(df):,})")
    print(f"  📉 Size reduction: {(1 - len(df_final)/len(df))*100:.1f}%")
    
    # 4. Initialize ChromaDB
    print("\n🗄️ Initializing ChromaDB...")
    collection = init_chroma_db(persist_directory="./chroma_db")
    
    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0:
        print(f"  ⚠️ Collection already has {existing_count:,} records")
        response = input("  Do you want to rebuild? This will delete existing data. (yes/no): ")
        if response.lower() != 'yes':
            print("  Aborted.")
            return
        
        # Delete and recreate
        collection.delete(ids=collection.get()['ids'])
        print("  🗑️ Cleared existing records")
    
    # 5. Add records to vector database
    print("\n🧮 Generating embeddings and adding to vector database...")
    print(f"  (This should take ~5-15 minutes for {len(df_final):,} records)")
    
    add_records_to_db(collection, df_final, batch_size=5000)
    
    # 6. Verify
    final_count = collection.count()
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 60)
    print("✅ Vector Database Build Complete!")
    print("=" * 60)
    print(f"  📊 Total Records: {final_count:,}")
    print(f"  ⏱️ Time Taken: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  💾 Location: ./chroma_db")
    print(f"  📦 Estimated Size: ~{final_count * 0.5 / 1000:.0f}MB")
    print("\n🎉 Your AI Data Analyst is now RAG-powered!")
    print("   You can now ask specific questions about any historical record.")
    print("\n💡 Note: Add 'chroma_db/' to .gitignore to avoid pushing to GitHub")

if __name__ == "__main__":
    main()
