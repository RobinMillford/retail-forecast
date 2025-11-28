"""
Pinecone Vector Database Client
Handles all interactions with Pinecone for vector storage and retrieval.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

class PineconeClient:
    """Wrapper for Pinecone operations."""
    
    def __init__(self):
        """Initialize Pinecone client."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "retail-sales")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Get or create index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment.split("-")[0] + "-" + self.environment.split("-")[1]
                )
            )
            print(f"✅ Index '{self.index_name}' created successfully")
        else:
            print(f"✅ Index '{self.index_name}' already exists")
    
    def create_record_text(self, row: Dict[str, Any]) -> str:
        """
        Create text representation of a sales record.
        
        Args:
            row: Dictionary with sales data
            
        Returns:
            Formatted text string
        """
        text = f"On {row['date']}, Store {row['store_nbr']}"
        
        if 'city' in row:
            text += f" in {row['city']}"
        
        text += f" sold {row['family']} with sales of ${row['sales']:.2f}"
        
        if 'onpromotion' in row and row['onpromotion']:
            text += " (on promotion)"
        
        if 'is_holiday' in row and row['is_holiday']:
            text += " during a holiday"
        
        return text
    
    def upsert_records(self, records: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert records to Pinecone.
        
        Args:
            records: List of record dictionaries
            batch_size: Number of records to process at once
        """
        total = len(records)
        print(f"Upserting {total:,} records to Pinecone...")
        
        for i in range(0, total, batch_size):
            batch = records[i:i+batch_size]
            
            # Prepare vectors
            vectors = []
            for record in batch:
                # Create text and embedding
                text = self.create_record_text(record)
                embedding = self.model.encode(text).tolist()
                
                # Create unique ID
                record_id = f"{record['date']}_{record['store_nbr']}_{record['family'].replace(' ', '_')}_{record.get('id', i)}"
                
                # Prepare metadata (Pinecone has limits on metadata size)
                metadata = {
                    'date': str(record['date']),
                    'store_nbr': int(record['store_nbr']),
                    'family': str(record['family']),
                    'sales': float(record['sales']),
                    'text': text  # Store original text for retrieval
                }
                
                # Add optional fields
                if 'city' in record:
                    metadata['city'] = str(record['city'])
                if 'state' in record:
                    metadata['state'] = str(record['state'])
                if 'onpromotion' in record:
                    metadata['onpromotion'] = int(record['onpromotion'])
                if 'is_holiday' in record:
                    metadata['is_holiday'] = int(record['is_holiday'])
                
                vectors.append({
                    'id': record_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert batch
            self.index.upsert(vectors=vectors)
            
            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {min(i + batch_size, total):,} / {total:,} records...")
        
        print(f"✅ Successfully upserted {total:,} records to Pinecone!")
    
    def query(self, query_text: str, top_k: int = 5, filter: Dict = None) -> List[Dict]:
        """
        Query Pinecone for similar records.
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of matching records with metadata
        """
        # Generate query embedding
        query_embedding = self.model.encode(query_text).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Format results
        matches = []
        for match in results['matches']:
            matches.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            })
        
        return matches
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness
        }
    
    def delete_all(self):
        """Delete all vectors from index (use with caution!)."""
        self.index.delete(delete_all=True)
        print("⚠️ All vectors deleted from index")


# Convenience function
def get_pinecone_client() -> PineconeClient:
    """Get initialized Pinecone client."""
    return PineconeClient()
