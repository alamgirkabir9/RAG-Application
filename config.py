import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Railway environment variables
    LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
    MODEL_NAME = "LLaMA 3 70B"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector database settings - use /tmp for Railway
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/tmp/vector_store")

    # RAG settings
    MAX_DOCUMENTS = 5
    SIMILARITY_THRESHOLD = 1.5
    
    # Railway specific settings
    PORT = int(os.getenv("PORT", 5000))
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
