"""
Configuration settings for the RAG Assistant
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
EMBEDDINGS_MODEL_CACHE = PROJECT_ROOT / "models"

# Create directories if they don't exist
DOCS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)
EMBEDDINGS_MODEL_CACHE.mkdir(exist_ok=True)

# Text Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOC = 100

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative: "BAAI/bge-small-en"

# Vector Database
VECTOR_DB_TYPE = "FAISS"  # Options: "FAISS", "ChromaDB"
TOP_K_RESULTS = 5

# LLM Configuration
LLM_PROVIDER = "groq"  # Options: "groq", "huggingface"
GROQ_MODEL = "mixtral-8x7b-32768"  # Options: "mixtral-8x7b-32768", "llama2-70b-4096"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# API Keys (will be loaded from environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Generation Parameters
MAX_TOKENS = 1000
TEMPERATURE = 0.1

# Web Search Fallback
ENABLE_WEB_SEARCH = True
WEB_SEARCH_THRESHOLD = 0.3  # If max similarity score is below this, use web search
