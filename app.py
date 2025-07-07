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

# Force rebuild the index to ensure consistency
document_store.rebuild_index_from_scratch()
print("Index rebuilt")

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
        # Save the file temporarily
        temp_path = os.path.join("data", file.filename)
        file.save(temp_path)
        
        # Process the file and add to document store
        document_store.add_document(temp_path)
        
        return redirect(url_for('index'))

@app.route('/api/generate', methods=['POST'])
async def api_generate():
    """API endpoint to generate text based on stored data"""
    data = request.json
    query = data.get('query', '')
    type = data.get('type', 'bio')  # bio, cover letter, etc.
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Generate response using RAG pipeline
    response = await rag_pipeline.generate(query, type)
    
    return jsonify({"response": response})

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    """Generate text based on a query and display results"""
    if request.method == 'POST':
        query = request.form.get('query', '')
        type = request.form.get('type', 'bio')
        
        if query:
            # Run the async function using asyncio to get the actual result
            response = asyncio.run(rag_pipeline.generate(query, type))
            return render_template('generate.html', query=query, response=response)
    
    return render_template('generate.html')

@app.route('/debug/documents', methods=['GET'])
def debug_documents():
    """Debug endpoint to view stored documents"""
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

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    app.run(debug=False, use_reloader=False)