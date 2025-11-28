"""
Daily Pinecone Update
Adds new sales records to Pinecone after nightly training.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pinecone_client import get_pinecone_client

def load_latest_data(days: int = 1) -> pd.DataFrame:
    """
    Load latest sales data from Kaggle.
    
    Args:
        days: Number of days to load (default: 1 for daily updates)
        
    Returns:
        DataFrame with latest sales data
    """
    print(f"\n📂 Loading last {days} day(s) of data...")
    
    # Load training data
    train_df = pd.read_csv('./data/train.csv', parse_dates=['date'])
    
    # Load store metadata
    stores_df = pd.read_csv('./data/stores.csv')
    
    # Merge
    df = train_df.merge(stores_df, on='store_nbr', how='left')
    
    # Filter to recent data
    cutoff_date = df['date'].max() - timedelta(days=days)
    df = df[df['date'] > cutoff_date]
    
    if len(df) == 0:
        print("  ⚠️ No new records found")
        return pd.DataFrame()
    
    print(f"  ✅ Loaded {len(df):,} new records from {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df

def prepare_records(df: pd.DataFrame) -> list:
    """
    Convert DataFrame to list of record dictionaries.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        List of record dictionaries
    """
    if len(df) == 0:
        return []
    
    print("\n🔄 Preparing records...")
    
    records = []
    for idx, row in df.iterrows():
        record = {
            'id': idx,
            'date': row['date'].date(),
            'store_nbr': row['store_nbr'],
            'family': row['family'],
            'sales': row['sales'],
            'onpromotion': row.get('onpromotion', 0)
        }
        
        # Add optional fields
        if 'city' in row and pd.notna(row['city']):
            record['city'] = row['city']
        if 'state' in row and pd.notna(row['state']):
            record['state'] = row['state']
        if 'type' in row and pd.notna(row['type']):
            record['type'] = row['type']
        
        records.append(record)
    
    print(f"  ✅ Prepared {len(records):,} records")
    return records

def main():
    """Main execution function."""
    print("=" * 60)
    print("🌲 Pinecone Daily Update")
    print("=" * 60)
    
    try:
        # Initialize Pinecone client
        print("\n🔗 Connecting to Pinecone...")
        client = get_pinecone_client()
        
        # Check current stats
        stats = client.get_stats()
        print(f"\n📊 Current Index Stats:")
        print(f"  Total vectors: {stats['total_vectors']:,}")
        
        # Load latest data
        df = load_latest_data(days=1)
        
        if len(df) == 0:
            print("\n✅ No new data to upload")
            return
        
        # Prepare records
        records = prepare_records(df)
        
        # Upsert to Pinecone
        print(f"\n📤 Uploading to Pinecone...")
        start_time = datetime.now()
        client.upsert_records(records, batch_size=100)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n⏱️  Upload completed in {elapsed:.1f}s")
        
        # Final stats
        stats = client.get_stats()
        print(f"\n📊 Updated Index Stats:")
        print(f"  Total vectors: {stats['total_vectors']:,}")
        print(f"  New vectors added: {len(records):,}")
        
        print("\n" + "=" * 60)
        print("✅ Daily update complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
