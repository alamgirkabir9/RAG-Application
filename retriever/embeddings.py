from sentence_transformers import SentenceTransformer
from config import Config

def get_embedding_model():
    """
    Load the embedding model
    
    Returns:
        SentenceTransformer: The embedding model
    """
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return model