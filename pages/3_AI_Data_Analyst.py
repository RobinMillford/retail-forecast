import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import plotly.express as px
from upstash_redis import Redis
import json
from datetime import datetime, timedelta
from utils import ui # Import shared UI

# --- UI SETUP ---
ui.setup_page(page_title="AI Data Analyst", page_icon="🤖")
load_dotenv()

# --- PAGE-SPECIFIC CSS (Chat bubbles and custom elements) ---
st.markdown(
    """
    <style>
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    /* Chat Messages */
    .user-msg { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 12px 0;
        margin-left: auto;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
        color: white;
        font-weight: 500;
    }
    
    .ai-msg { 
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 16px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 12px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: slideInLeft 0.3s ease-out;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True
)

# Note: Gemini is configured as fallback in get_ai_response_rag()

# --- DATA LOADING (RAG-POWERED WITH HF AUTO-DOWNLOAD) ---
@st.cache_resource
def load_vector_db():
    """Connect to Pinecone vector database for RAG."""
    try:
        from utils.pinecone_client import get_pinecone_client
        
        # Check for Pinecone credentials
        if not os.getenv("PINECONE_API_KEY"):
            st.warning("""
            ⚠️ **Pinecone not configured**
            
            To use the vector database:
            1. Set `PINECONE_API_KEY` in your `.env` file
            2. Set `PINECONE_ENVIRONMENT` (e.g., `us-east-1-aws`)
            3. Set `PINECONE_INDEX_NAME` (e.g., `retail-sales`)
            
            Or run locally: `python scripts/pinecone_initial_load.py`
            """)
            return None
        
        # Connect to Pinecone
        with st.spinner("Connecting to Pinecone..."):
            client = get_pinecone_client()
            stats = client.get_stats()
        
        # Check if populated
        if stats['total_vectors'] == 0:
            st.warning("⚠️ Pinecone index is empty.")
            st.info("""
            **First-time setup required:**
            
            Run this command to upload data:
            ```bash
            python scripts/pinecone_initial_load.py
            ```
            
            This will upload the last 6 months of sales data to Pinecone.
            """)
            return None
        
        st.success(f"✅ Connected to Pinecone with {stats['total_vectors']:,} vectors")
        return client
    except Exception as e:
        st.error(f"❌ Error connecting to Pinecone: {e}")
        return None

# Live data is now handled by GitHub Actions (incremental_build.py)
# Removed unused functions: load_live_data() and add_live_data_to_vector_db()

# --- RAG LOGIC ---
def parse_query_filters(prompt):
    """Extract metadata filters from user query with error handling.
    
    Note: Only extracts store_nbr and date for exact filtering.
    Product families are handled by semantic search since they have variations
    (e.g., GROCERY I, GROCERY II, etc.)
    """
    import re
    filters = {}
    
    try:
        # Extract store number (exact match works)
        store_match = re.search(r'store\s+(\d+)', prompt.lower())
        if store_match:
            filters['store_nbr'] = int(store_match.group(1))
        
        # Extract date (exact match works)
        date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', prompt)
        if date_match:
            filters['date'] = date_match.group(0)
        
        # NOTE: Product family filtering removed - semantic search handles it better
        # Reason: Families have variations (GROCERY I, GROCERY II, BEVERAGES, etc.)
        # and exact matching fails. Semantic search is more flexible.
        
    except Exception as e:
        print(f"Filter parsing error: {e}")
        return None
    
    return filters if filters else None

def get_ai_response_rag(prompt, pinecone_client):
    """
    RAG-powered AI response using Pinecone for retrieval and Groq for generation.
    
    Args:
        prompt: User's question
        pinecone_client: Pinecone client instance
        
    Returns:
        AI-generated answer based on retrieved records
    """
    # Check for API key (try Groq first, fallback to Gemini)
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not groq_api_key and not gemini_api_key:
        return "⚠️ No API key found. Please set GROQ_API_KEY or GEMINI_API_KEY in your .env file."
    
    if pinecone_client is None:
        return "❌ Vector database not available. Please run `python scripts/pinecone_initial_load.py` to create it."
    
    try:
        # 1. Parse query for filters
        filters = parse_query_filters(prompt)
        
        # 2. Retrieve relevant records from Pinecone (with filters if available)
        relevant_records = pinecone_client.query(prompt, top_k=30, filter=filters)
        
        if not relevant_records:
            filter_msg = f" with filters {filters}" if filters else ""
            
            # DEBUG: Try without filters to see what's available
            if filters:
                st.warning(f"No records found with filters: {filters}")
                st.info("🔍 **Trying without filters to see available data...**")
                
                # Query without filters
                relevant_records_unfiltered = pinecone_client.query(prompt, top_k=10, filter=None)
                
                if relevant_records_unfiltered:
                    st.success(f"Found {len(relevant_records_unfiltered)} records without filters. Here's a sample:")
                    
                    # Show sample of what's available
                    sample_stores = set()
                    sample_families = set()
                    for rec in relevant_records_unfiltered[:5]:
                        if 'store_nbr' in rec['metadata']:
                            sample_stores.add(rec['metadata']['store_nbr'])
                        if 'family' in rec['metadata']:
                            sample_families.add(rec['metadata']['family'])
                    
                    st.write(f"**Available stores in results:** {sorted(sample_stores)}")
                    st.write(f"**Available families in results:** {sorted(sample_families)}")
                    st.write(f"**Sample record:** {relevant_records_unfiltered[0]['metadata']['text'][:200]}...")
                    
                    return f"I couldn't find records matching your exact filters ({filters}), but I found similar data. Try asking about stores {sorted(sample_stores)} or product families like {', '.join(sorted(sample_families))}."
            
            return f"I couldn't find any relevant records for your question{filter_msg}. Try rephrasing or asking about a different topic."
        
        # 3. Format retrieved records as context
        context = "# RETRIEVED SALES RECORDS\n\n"
        if filters:
            context += f"Filters applied: {filters}\n\n"
        context += f"Found {len(relevant_records)} relevant records:\n\n"
        
        for i, record in enumerate(relevant_records[:20], 1):  # Limit to top 20 for context window
            context += f"**Record {i}:**\n"
            context += f"{record['metadata']['text']}\n\n"
        
        # 4. Create prompt
        full_prompt = f"""You are an expert Retail Data Analyst with access to a comprehensive sales database.

RETRIEVED DATA:
{context}

USER QUESTION: {prompt}

INSTRUCTIONS:
1. Analyze the retrieved records carefully.
2. Answer the user's question based ONLY on the data provided above.
3. Be specific - cite dates, stores, sales figures, and other details.
4. If the data doesn't fully answer the question, say so and provide what you can.
5. Use a professional, concise tone.
6. Format numbers with commas (e.g., $1,234.56).

ANSWER:
"""
        
        # 5. Generate response (try Groq first, fallback to Gemini)
        if groq_api_key:
            # Use Groq (Llama 3.3 70B - Fast and Free!)
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fast, powerful, free
                messages=[
                    {"role": "system", "content": "You are an expert Retail Data Analyst."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        else:
            # Fallback to Gemini
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(full_prompt)
            return response.text
        
    except Exception as e:
        return f"AI Error: {e}"

# --- HEADER (will be updated after DB loads) ---
header_placeholder = st.empty()
header_placeholder.markdown("""
<div class="main-header">
    <h1 class="main-title">🤖 AI Data Analyst</h1>
    <p class="subtitle">RAG-Powered Semantic Search • Loading... • Llama 3.3 70B</p>
</div>
""", unsafe_allow_html=True)

# Status Bar (Dynamic)
col_status1, col_status2, col_status3, col_status4 = st.columns(4)
with col_status1:
    # Will be updated after loading
    st.empty()
with col_status2:
    groq_key = os.getenv("GROQ_API_KEY")
    model_name = "Groq Llama" if groq_key else "Gemini"
    st.metric("🤖 AI Model", model_name, delta="Active")
with col_status3:
    st.metric("⚡ Status", "Ready", delta="Online")
with col_status4:
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# --- LOAD VECTOR DATABASE ---
with st.spinner("Connecting to Pinecone vector database..."):
    pinecone_client = load_vector_db()
    
    # Update status metric and header with actual count
    if pinecone_client:
        stats = pinecone_client.get_stats()
        record_count = stats['total_vectors']
        col_status1.metric("🗄️ Vector DB", "Connected", delta=f"{record_count:,} vectors")
        
        # Update header with actual count
        header_placeholder.markdown(f"""
<div class="main-header">
    <h1 class="main-title">🤖 AI Data Analyst</h1>
    <p class="subtitle">RAG-Powered Pinecone • {record_count:,} Vectors • Llama 3.3 70B</p>
</div>
""", unsafe_allow_html=True)
    else:
        col_status1.metric("🗄️ Vector DB", "Not Connected", delta="Setup required")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Container
chat_container = st.container()

with chat_container:
    # Display History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f'<div class="user-msg">👤 <b>You:</b> {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ai-msg">🤖 <b>Analyst:</b> {content}</div>', unsafe_allow_html=True)

# Input
prompt = st.chat_input("Ask me anything (e.g., 'What was the sales for GROCERY in Store 5 on 2017-12-25?')")

if prompt:
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Get AI Response using RAG
    with st.spinner("Searching Pinecone and analyzing..."):
        response = get_ai_response_rag(prompt, pinecone_client)
    
    # 3. Add AI Message
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to display new messages
    st.rerun()

# Info: How RAG Works
with st.expander("ℹ️ How This Works (RAG Architecture)", expanded=False):
    vector_count = pinecone_client.get_stats()['total_vectors'] if pinecone_client else 0
    st.markdown(f"""
    <div class="info-card">
    
    **This AI uses Retrieval Augmented Generation (RAG):**
    
    1. **🔍 Your Question** → Converted to a vector embedding
    2. **🎯 Smart Filtering** → Extracts store, date, product filters
    3. **📊 Semantic Search** → Pinecone finds the most relevant sales records
    4. **📝 Context** → Top 20-30 records are sent to Llama 3.3 70B
    5. **💬 Answer** → AI generates a response based on retrieved data
    
    ---
    
    **✨ Features:**
    - ✅ Query **any** of {vector_count:,} vectors
    - ✅ Intelligent metadata filtering
    - ✅ Semantic search for fuzzy queries
    - ✅ Powered by Groq (fast & free!)
    - ✅ Cloud-hosted on Pinecone
    
    **📦 Database:** {vector_count:,} vectors on Pinecone
    
    </div>
    """, unsafe_allow_html=True)

