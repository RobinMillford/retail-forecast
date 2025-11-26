"""
Vector Database Manager for RAG-powered AI Data Analyst
Uses ChromaDB for semantic search over sales records
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

# Initialize embedding model (cached globally)
_embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model to save memory."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
    return _embedding_model

def init_chroma_db(persist_directory="./chroma_db"):
    """
    Initialize or load ChromaDB collection.
    
    Args:
        persist_directory: Where to store the vector database
        
    Returns:
        ChromaDB collection
    """
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="sales_records",
        metadata={"description": "Retail sales data with embeddings"}
    )
    
    return collection

def create_record_text(row: pd.Series) -> str:
    """
    Convert a sales record to a text string for embedding.
    
    Example output:
    "Date: 2017-12-25 (Monday), Store: 5 in Quito, Pichincha (Type A), 
     Product: GROCERY, Sales: $1,234.56, Promotion: Yes, Holiday: Christmas, 
     Oil Price: $45.23"
    """
    date_str = row['date'].strftime('%Y-%m-%d (%A)')
    
    # Store info
    store_info = f"Store: {row['store_nbr']}"
    if 'city' in row and pd.notna(row['city']):
        store_info += f" in {row['city']}, {row.get('state', 'Unknown')} (Type {row.get('type', 'Unknown')})"
    
    # Sales info
    sales_info = f"Product: {row['family']}, Sales: ${row['sales']:.2f}"
    
    # Promotion
    promo = "Promotion: Yes" if row.get('onpromotion', 0) == 1 else "Promotion: No"
    
    # Holiday
    holiday = "Holiday: Yes" if row.get('is_holiday', 0) == 1 else "Holiday: No"
    
    # Oil price
    oil = f"Oil Price: ${row['dcoilwtico']:.2f}" if 'dcoilwtico' in row and pd.notna(row['dcoilwtico']) else ""
    
    text = f"Date: {date_str}, {store_info}, {sales_info}, {promo}, {holiday}"
    if oil:
        text += f", {oil}"
    
    return text

def add_records_to_db(collection, df: pd.DataFrame, batch_size: int = 1000):
    """
    Add sales records to ChromaDB in batches.
    
    Args:
        collection: ChromaDB collection
        df: DataFrame with sales records
        batch_size: Number of records to process at once
    """
    model = get_embedding_model()
    
    total_records = len(df)
    print(f"Adding {total_records:,} records to vector database...")
    
    # Get the current max ID to avoid collisions
    try:
        existing_count = collection.count()
    except:
        existing_count = 0
    
    for i in range(0, total_records, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Create text representations
        texts = [create_record_text(row) for _, row in batch.iterrows()]
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        
        # Prepare metadata
        metadatas = []
        ids = []
        for idx, row in batch.iterrows():
            metadata = {
                'date': str(row['date'].date()),
                'store_nbr': int(row['store_nbr']),
                'family': str(row['family']),
                'sales': float(row['sales']),
            }
            
            # Add optional fields
            if 'city' in row and pd.notna(row['city']):
                metadata['city'] = str(row['city'])
            if 'state' in row and pd.notna(row['state']):
                metadata['state'] = str(row['state'])
            if 'onpromotion' in row:
                metadata['onpromotion'] = int(row['onpromotion'])
            if 'is_holiday' in row:
                metadata['is_holiday'] = int(row['is_holiday'])
            
            metadatas.append(metadata)
            
            # Create unique ID based on data (date_store_family)
            # This ensures each record has a truly unique ID
            unique_id = f"{row['date'].date()}_{row['store_nbr']}_{row['family'].replace(' ', '_')}"
            ids.append(unique_id)
        
        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        if (i + batch_size) % 10000 == 0:
            print(f"  Processed {i + batch_size:,} / {total_records:,} records...")
    
    print(f"✅ Successfully added {total_records:,} records to vector database!")

def query_records(collection, question: str, top_k: int = 20, filters: Dict = None) -> List[Dict]:
    """
    Query the vector database for relevant records.
    
    Args:
        collection: ChromaDB collection
        question: User's question
        top_k: Number of results to return
        filters: Optional metadata filters (e.g., {'store_nbr': 5, 'date': '2017-12-25'})
        
    Returns:
        List of relevant records with metadata
    """
    model = get_embedding_model()
    
    # Generate embedding for the question
    question_embedding = model.encode([question])[0].tolist()
    
    # Format filters for ChromaDB (use $eq for equality)
    where_clause = None
    if filters:
        if len(filters) == 1:
            # Single filter with $eq operator
            key, value = list(filters.items())[0]
            where_clause = {key: {"$eq": value}}
        else:
            # Multiple filters - use $and with $eq operators
            where_clause = {
                "$and": [
                    {key: {"$eq": value}} for key, value in filters.items()
                ]
            }
    
    # Query ChromaDB
    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
    except Exception as e:
        print(f"Query error: {e}")
        # Fallback: query without filters
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
    
    # Format results
    records = []
    if results and 'ids' in results and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            record = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            }
            records.append(record)
    
    return records
