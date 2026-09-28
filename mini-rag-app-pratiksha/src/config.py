import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# RAG Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Local lightweight sentence-transformer
LLM_MODEL_NAME = "gemini-3.8-flash"        # Gemini model
VECTOR_STORE_DIR = "./chroma_db"
DATA_DIR = "./data"

# Text Splitter Settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 3
