import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
import uuid
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from retriever.embeddings import get_embedding_model
from config import Config

class DocumentStore:
    """Vector store for document storage and retrieval"""
    
    def __init__(self, vector_db_path: Optional[str] = None):
        """Initialize the document store"""
        self.vector_db_path = vector_db_path or Config.VECTOR_DB_PATH
        print(f"Using vector DB path: {self.vector_db_path}")
        
        self.embeddings = get_embedding_model()
        print("Embedding model loaded")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Create directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Check if index exists, otherwise create it
        self.index_path = os.path.join(self.vector_db_path, "faiss_index")
        self.documents_path = os.path.join(self.vector_db_path, "documents.json")
        
        print(f"Index path: {self.index_path}")
        print(f"Documents path: {self.documents_path}")
        
        # Load or create index
        if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
            print("Found existing index and documents, loading...")
            self.load()
        else:
            print("No existing index found, initializing empty one...")
            # Initialize an empty index
            self.documents = {}
            self.document_embeddings = {}
            self.initialize_index()
            
    def initialize_index(self):
        """Initialize an empty FAISS index"""
        # Get embedding dimension from the model
        test_embedding = self.embeddings.encode("test")
        dimension = len(test_embedding)
        
        # Create empty index
        self.index = faiss.IndexFlatL2(dimension)
        self.save()
    
    def add_text(self, content: str, title: str = "Untitled") -> str:
        """
        Add text content to the document store
        
        Args:
            content (str): The text content to add
            title (str): Title for the content
            
        Returns:
            str: Document ID
        """
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Store document metadata
        self.documents[doc_id] = {
            "title": title,
            "chunks": chunks,
            "type": "text"
        }
        
        # Compute and store embeddings for each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.encode(chunk)
            chunk_id = f"{doc_id}_{i}"
            self.document_embeddings[chunk_id] = {
                "doc_id": doc_id,
                "chunk_index": i
            }
            chunk_embeddings.append(embedding)
        
        # Add embeddings to FAISS index
        if chunk_embeddings:
            self.index.add(np.array(chunk_embeddings, dtype=np.float32))
            self.save()
        
        return doc_id
    
    def add_document(self, file_path: str) -> str:
        """
        Process and add a document file to the store
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Document ID
        """
        # Determine file type and use appropriate loader
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Extract text from documents
        content = "\n\n".join([doc.page_content for doc in docs])
        title = os.path.basename(file_path)
        
        # Add text to document store
        return self.add_text(content, title)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: List of document chunks with metadata
        """
        # Check if there are any documents first
        if not self.documents:
            print("No documents in store during search")
            return []
            
        # Print debug information
        print(f"Searching for: {query}")
        print(f"Document count: {len(self.documents)}")
        print(f"Document embeddings count: {len(self.document_embeddings)}")
        
        # Encode the query
        query_vector = self.embeddings.encode(query)
        query_vector = np.array([query_vector], dtype=np.float32)
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        print(f"Search returned {len(indices[0])} results")
        print(f"Indices: {indices[0]}")
        print(f"Distances: {distances[0]}")
        
        results = []
        for i, idx in enumerate(indices[0]):
            # Skip invalid indices
            if idx == -1:
                continue
                
            # Skip results with distance above threshold - TEMPORARILY DISABLED FOR DEBUGGING
            # if distances[0][i] > Config.SIMILARITY_THRESHOLD:
            #     print(f"Skipping result with distance {distances[0][i]} (above threshold {Config.SIMILARITY_THRESHOLD})")
            #     continue
            print(f"Processing result with distance {distances[0][i]}")
                
            # Find the corresponding chunk ID
            chunk_ids = list(self.document_embeddings.keys())
            if idx >= len(chunk_ids):
                print(f"Index {idx} out of range for chunk_ids (len: {len(chunk_ids)})")
                continue
                
            chunk_id = chunk_ids[idx]
            chunk_info = self.document_embeddings[chunk_id]
            doc_id = chunk_info["doc_id"]
            chunk_index = chunk_info["chunk_index"]
            
            # Get document content
            if doc_id not in self.documents:
                print(f"Document ID {doc_id} not found in documents")
                continue
                
            document = self.documents[doc_id]
            if chunk_index >= len(document["chunks"]):
                print(f"Chunk index {chunk_index} out of range for document {doc_id}")
                continue
                
            chunk_content = document["chunks"][chunk_index]
            
            print(f"Found relevant chunk: {chunk_content[:50]}...")
            
            results.append({
                "content": chunk_content,
                "title": document["title"],
                "similarity": float(1 - distances[0][i] / 2),  # Normalize similarity score
                "doc_id": doc_id
            })
        
        print(f"Returning {len(results)} results")
        return results
    def save(self):
        """Save the index and documents to disk"""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save documents and mappings
        data = {
            "documents": self.documents,
            "document_embeddings": self.document_embeddings
        }
        with open(self.documents_path, 'w') as f:
            json.dump(data, f)
    
    def load(self):
        """Load the index and documents from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load documents and mappings
            with open(self.documents_path, 'r') as f:
                data = json.load(f)
                self.documents = data.get("documents", {})
                self.document_embeddings = data.get("document_embeddings", {})
                
            print(f"Loaded {len(self.documents)} documents and {len(self.document_embeddings)} embeddings")
            
            # Verify document structure
            for doc_id, doc in self.documents.items():
                if "chunks" not in doc:
                    print(f"Warning: Document {doc_id} missing 'chunks' field")
                elif not doc["chunks"]:
                    print(f"Warning: Document {doc_id} has empty 'chunks' list")
                
            # Verify embedding-document relationships
            for chunk_id, chunk_info in self.document_embeddings.items():
                doc_id = chunk_info.get("doc_id")
                if doc_id not in self.documents:
                    print(f"Warning: Embedding {chunk_id} refers to non-existent document {doc_id}")
                    continue
                    
                chunk_index = chunk_info.get("chunk_index")
                if chunk_index is None:
                    print(f"Warning: Embedding {chunk_id} missing 'chunk_index'")
                    continue
                    
                doc = self.documents[doc_id]
                if "chunks" not in doc or chunk_index >= len(doc["chunks"]):
                    print(f"Warning: Embedding {chunk_id} refers to non-existent chunk {chunk_index} in document {doc_id}")
        
        except Exception as e:
            print(f"Error loading document store: {e}")
            # Initialize empty collections
            self.documents = {}
            self.document_embeddings = {}
            self.initialize_index()
            
    def rebuild_index(self):
        """Rebuild the index from all documents"""
        # Get embedding dimension
        test_embedding = self.embeddings.encode("test")
        dimension = len(test_embedding)
        
        # Create a new index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Re-embed and add all chunks
        all_embeddings = []
        
        for doc_id, doc_info in self.documents.items():
            chunks = doc_info.get("chunks", [])
            for chunk in chunks:
                embedding = self.embeddings.encode(chunk)
                all_embeddings.append(embedding)
        
        if all_embeddings:
            self.index.add(np.array(all_embeddings, dtype=np.float32))
            
        self.save()

    def load_from_json(self, json_data):
        """Load documents from provided JSON data"""
        self.documents = json_data.get("documents", {})
        self.document_embeddings = json_data.get("document_embeddings", {})
        
        # Rebuild the index
        self.rebuild_index()

    def rebuild_index_from_scratch(self):
        """Completely rebuild the index from the documents"""
        print("Rebuilding search index from scratch...")
        
        # Get embedding dimension
        test_embedding = self.embeddings.encode("test")
        dimension = len(test_embedding)
        
        # Create a new index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Track mappings between index positions and document chunks
        self.document_embeddings = {}
        current_idx = 0
        
        # Re-embed and add all chunks
        all_embeddings = []
        
        for doc_id, doc_info in self.documents.items():
            chunks = doc_info.get("chunks", [])
            print(f"Processing document {doc_id} with {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.encode(chunk)
                all_embeddings.append(embedding)
                
                # Store mapping
                chunk_id = f"{doc_id}_{i}"
                self.document_embeddings[chunk_id] = {
                    "doc_id": doc_id,
                    "chunk_index": i
                }
                current_idx += 1
        
        # Add all embeddings to index at once
        if all_embeddings:
            print(f"Adding {len(all_embeddings)} embeddings to index")
            self.index.add(np.array(all_embeddings, dtype=np.float32))
        else:
            print("No embeddings to add to index")
            
        self.save()
        print("Index rebuild complete")