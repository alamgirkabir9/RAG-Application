# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows you to upload documents and generate professional bios, cover letters, and other content based on your stored information.

## Features

- **Document Upload**: Support for PDF and TXT files
- **Text Input**: Direct text input for adding information
- **RAG Pipeline**: Retrieve relevant information and generate responses
- **Multiple Generation Types**: Bio, cover letter, and general responses
- **Vector Search**: Efficient document retrieval using FAISS

## Setup
## 🌐 Live Demo  
**Deployed on huggingface**: [(https://huggingface.co/spaces/Alamgirapi/Professional)]
### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
LLAMA_API_KEY=your_groq_api_key_here
SECRET_KEY=your_secret_key_here
VECTOR_DB_PATH=data/vector_store
DEBUG=False
```

### Local Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables
4. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. **Add Data**: Upload documents or add text directly through the web interface
2. **Generate Content**: Use the generate page to create bios, cover letters, or get general responses
3. **Debug**: View stored documents and their chunks in the debug section

## API Endpoints

- `POST /api/generate`: Generate content based on query and type
- `GET /debug/documents`: View stored documents (for debugging)
- `GET /health`: Health check endpoint

## Deployment

This application is designed to run on Hugging Face Spaces. The configuration includes:
- Dockerfile for containerization
- Port configuration for Hugging Face Spaces (7860)
- Environment variable support
- Proper error handling and logging

## Technical Details

- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Groq API with LLaMA 3 70B
- **Framework**: Flask for web interface
- **Document Processing**: LangChain for document loading and splitting

## License

This project is open source and available under the MIT License.
