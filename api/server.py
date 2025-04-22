from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths to your model files - adjust these to match your actual file paths
EMBEDDINGS_PATH = 'embeddings.pkl'
CHUNKS_PATH = 'processed_chunks.pkl'

# Load model and data
model = None
chunks_df = None
embeddings = None
index = None

def load_model():
    global model, chunks_df, embeddings, index
    
    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load chunks dataframe
    with open(CHUNKS_PATH, 'rb') as f:
        chunks_df = pickle.load(f)
    
    # Load embeddings
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print("Model and data loaded successfully")

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    property_data = data.get('property_data', {})
    
    # Enhance query with property data if available
    enhanced_query = user_query
    if property_data:
        property_context = f"For a {property_data.get('beds', '')} bedroom, {property_data.get('baths', '')} bathroom home of {property_data.get('sqft', '')} square feet, priced at {property_data.get('price', '')}, located at {property_data.get('address', '')}: "
        enhanced_query = property_context + user_query
    
    # Embed the query
    query_embedding = model.encode([enhanced_query])
    
    # Search for relevant chunks
    k = 5  # Number of chunks to retrieve
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Get the results
    result_chunks = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks_df):
            chunk = chunks_df.iloc[idx]
            result_chunks.append({
                'source_file': chunk['source_file'],
                'text': chunk['text'],
                'distance': float(distances[0][i])
            })
    
    # Format response
    response = "Based on the retrieved documents:\n\n"
    for i, chunk in enumerate(result_chunks):
        response += f"From {chunk['source_file']}:\n"
        response += f"{chunk['text'][:300]}...\n\n"
    
    # Add property context
    if property_data:
        property_summary = f"\nProperty details: {property_data.get('beds', 'N/A')} bed, {property_data.get('baths', 'N/A')} bath, {property_data.get('sqft', 'N/A')} sqft, {property_data.get('price', 'N/A')}"
        response += property_summary
    
    # Add sources
    sources = list(set(chunk['source_file'] for chunk in result_chunks))
    response += "\nSources: " + ", ".join(sources)
    
    return jsonify({
        'answer': response
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
