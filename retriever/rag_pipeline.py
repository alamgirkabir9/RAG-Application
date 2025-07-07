from typing import List, Dict, Any
from retriever.document_store import DocumentStore
from config import Config

class RAGPipeline:
    """RAG pipeline for generating responses based on retrieved documents"""
    
    def __init__(self, document_store: DocumentStore, llm: Any):
        """
        Initialize the RAG pipeline
        
        Args:
            document_store (DocumentStore): Document store for retrieval
            llm (Any): Language model
        """
        self.document_store = document_store
        self.llm = llm
        
        # Define templates for different generation types
        self.templates = {
            "bio": (
                "Using the following information about a person, create a professional bio for Upwork:\n\n"
                "Information:\n{context}\n\n"
                "Query: {query}\n\n"
                "Create a well-structured professional profile that highlights their skills, experience, and achievements. "
                "The bio should be engaging, professional, and optimized for Upwork. "
                "Only use information that is provided in the context."
            ),
            "cover_letter": (
                "Using the following information about a person, create a personalized cover letter:\n\n"
                "Information:\n{context}\n\n"
                "Job description or query: {query}\n\n"
                "Create a professional cover letter that connects their experience to the job requirements. "
                "Highlight relevant skills and achievements from the provided information. "
                "Only use information that is provided in the context."
            ),
            "general": (
                "Based on the following information:\n\n"
                "{context}\n\n"
                "Please respond to this question or request: {query}\n\n"
                "Only use information that is provided in the context. If the information isn't sufficient, "
                "state that clearly rather than making up information."
            )
        }
    
    async def generate(self, query: str, type: str = "bio") -> str:
        """
        Generate a response based on stored documents and query
        
        Args:
            query (str): Query or request
            type (str): Type of generation (bio, cover_letter, general)
            
        Returns:
            str: Generated response
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.document_store.search(
                query, 
                top_k=Config.MAX_DOCUMENTS
            )
            
            if not retrieved_docs:
                return "I couldn't find any relevant information to answer that query. Please add more data to the system."
            
            # Combine retrieved documents into context
            context = "\n\n".join([f"Document: {doc['title']}\nContent: {doc['content']}" for doc in retrieved_docs])
            
            # Select appropriate template
            template = self.templates.get(type, self.templates["general"])
            
            # Format prompt with context and query
            prompt = template.format(context=context, query=query)
            
            # Call LLM
            response = await self.llm(prompt)
            
            # Extract text from response
            if isinstance(response, dict):
                try:
                    response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not response_text:
                        response_text = "Failed to generate a response. Please try again."
                except (IndexError, KeyError) as e:
                    print(f"Error extracting response: {e}")
                    response_text = "Error processing the response from language model."
            else:
                response_text = str(response)
            
            return response_text
            
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"