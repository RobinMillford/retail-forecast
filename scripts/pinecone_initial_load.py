"""
Initial Pinecone Data Load
Uploads last 6 months of historical sales data to Pinecone.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pinecone_client import get_pinecone_client

def load_recent_data(max_records: int = 500000) -> pd.DataFrame:
    """
    Load recent sales data (limited to max_records for Pinecone free tier).
    
    Args:
        max_records: Maximum number of records to load (default: 500K for free tier)
        
    Returns:
        DataFrame with recent sales data
    """
    print(f"\n📂 Loading up to {max_records:,} most recent records...")
    
    # Load training data
    train_df = pd.read_csv('./data/train.csv', parse_dates=['date'])
    
    # Load store metadata
    stores_df = pd.read_csv('./data/stores.csv')
    
    # Merge
    df = train_df.merge(stores_df, on='store_nbr', how='left')
    
    # Sort by date (most recent first) and take the last N records
    df = df.sort_values('date', ascending=False).head(max_records)
    df = df.sort_values('date').reset_index(drop=True)  # Re-sort chronologically
    
    print(f"  ✅ Loaded {len(df):,} records from {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df

def prepare_records(df: pd.DataFrame) -> list:
    """
    Convert DataFrame to list of record dictionaries.
    
    Args:
        df: Sales DataFrame
        
    Returns:
        List of record dictionaries
    """
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
    print("🌲 Pinecone Initial Data Load")
    print("=" * 60)
    
    try:
        # Initialize Pinecone client
        print("\n🔗 Connecting to Pinecone...")
        client = get_pinecone_client()
        
        # Check current stats
        stats = client.get_stats()
        print(f"\n📊 Current Index Stats:")
        print(f"  Total vectors: {stats['total_vectors']:,}")
        print(f"  Dimension: {stats['dimension']}")
        
        # Load data (500K records for free tier)
        df = load_recent_data(max_records=500000)
        
        # Prepare records
        records = prepare_records(df)
        
        # Upsert to Pinecone
        print(f"\n📤 Uploading to Pinecone...")
        start_time = datetime.now()
        client.upsert_records(records, batch_size=100)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n⏱️  Upload completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Final stats
        stats = client.get_stats()
        print(f"\n📊 Final Index Stats:")
        print(f"  Total vectors: {stats['total_vectors']:,}")
        print(f"  Index fullness: {stats['index_fullness']:.2%}")
        
        print("\n" + "=" * 60)
        print("✅ Initial load complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
