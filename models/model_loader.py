import os
import httpx
from config import Config

def load_llm(api_key=None):
    """
    Load the LLM client for generating responses
    
    Args:
        api_key (str): API key for the LLM service
        
    Returns:
        function: Async function to fetch responses from the model
    """
    if not api_key:
        api_key = os.getenv("LLAMA_API_KEY")
        
    if not api_key:
        raise ValueError("API key is required. Set LLAMA_API_KEY in the environment or pass it directly.")
    
    # Groq API endpoint for LLaMA 3 70B
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    model_name = "llama3-70b-8192"
    
    async def fetch_model_response(query: str):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_message = f"API Error: {response.status_code} - {response.text}"
                print(error_message)
                return {"error": "Failed to fetch response from model", "details": error_message}
        except Exception as e:
            print(f"Exception in fetch_model_response: {str(e)}")
            return {"error": "Exception calling LLM API", "details": str(e)}
    
    return fetch_model_response