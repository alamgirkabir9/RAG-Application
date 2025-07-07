import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
    MODEL_NAME = "LLaMA 3 70B"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector database settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_store")

    # RAG settings
    MAX_DOCUMENTS = 5  # Maximum number of documents to retrieve
    SIMILARITY_THRESHOLD = 1.5  # Modified threshold to be more lenient

