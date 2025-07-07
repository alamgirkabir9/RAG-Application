import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import asyncio
from retriever.document_store import DocumentStore
from retriever.rag_pipeline import RAGPipeline
from models.model_loader import load_llm
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
document_store = DocumentStore()
print("Document store initialized")

# Only rebuild index if it doesn't exist (for performance)
if not document_store.index_exists():
    document_store.rebuild_index_from_scratch()
    print("Index rebuilt")
else:
    print("Index already exists, skipping rebuild")

llm = load_llm(api_key=app.config["LLAMA_API_KEY"])
print("LLM loaded")

rag_pipeline = RAGPipeline(document_store, llm)
print("RAG pipeline initialized")

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    """Add data to the document store"""
    if request.method == 'POST':
        content = request.form.get('content')
        title = request.form.get('title', 'Untitled')
        
        # Save content as a document
        if content:
            document_store.add_text(title=title, content=content)
            return redirect(url_for('index'))
    
    return render_template('add_data.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Upload and process a file (PDF, TXT, etc.)"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        # Use /tmp directory for temporary files (works on Render free tier)
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        try:
            # Process the file and add to document store
            document_store.add_document(temp_path)
            return redirect(url_for('index'))
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.route('/api/generate', methods=['POST'])
async def api_generate():
    """API endpoint to generate text based on stored data"""
    data = request.json
    query = data.get('query', '')
    type = data.get('type', 'bio')  # bio, cover letter, etc.
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        # Generate response using RAG pipeline
        response = await rag_pipeline.generate(query, type)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """Generate text based on a query and display results"""
    if request.method == 'POST':
        query = request.form.get('query', '')
        type = request.form.get('type', 'bio')
        
        if query:
            try:
                # Run the async function using asyncio to get the actual result
                response = asyncio.run(rag_pipeline.generate(query, type))
                return render_template('generate.html', query=query, response=response)
            except Exception as e:
                return render_template('generate.html', error=f"Error: {str(e)}")
    
    return render_template('generate.html')

@app.route('/debug/documents', methods=['GET'])
def debug_documents():
    """Debug endpoint to view stored documents"""
    try:
        doc_count = len(document_store.documents)
        chunk_count = sum(len(doc.get('chunks', [])) for doc in document_store.documents.values())
        
        docs_summary = []
        for doc_id, doc in document_store.documents.items():
            docs_summary.append({
                "id": doc_id,
                "title": doc.get("title", "Untitled"),
                "chunks": len(doc.get("chunks", [])),
                "first_chunk_preview": doc.get("chunks", [""])[0][:100] + "..." if doc.get("chunks") else ""
            })
        
        return render_template(
            'debug.html', 
            doc_count=doc_count,
            chunk_count=chunk_count,
            docs=docs_summary
        )
    except Exception as e:
        return render_template('debug.html', error=f"Error: {str(e)}")

@app.route('/healthz')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "service": "RAG Application"})

if __name__ == '__main__':
    # Create data directory if it doesn't exist (use /tmp for Render free tier)
    data_dir = os.getenv("DATA_DIR", "/tmp/data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Run with appropriate settings for production
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
