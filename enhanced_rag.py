import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re


class EnhancedRealEstateRAG:
    def __init__(self,
                 text_chunks_path="processed_chunks.pkl",
                 structured_data_path="processed_structured_data.pkl",
                 embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the enhanced RAG system with document chunks and embedding model.

        Args:
            text_chunks_path: Path to the pickle file with processed text chunks
            structured_data_path: Path to the pickle file with processed structured data
            embedding_model: SentenceTransformer model to use for embeddings
        """
        self.text_chunks_path = text_chunks_path
        self.structured_data_path = structured_data_path
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)

        self.text_index = None
        self.text_chunks_df = None
        self.text_embeddings = None

        self.structured_data = None
        self.structured_data_descriptions = None
        self.structured_data_index = None
        self.structured_data_embeddings = None

        # Load data if it exists
        self.load_data()

    def load_data(self):
        """Load document chunks and structured data from pickle files."""
        # Load text chunks
        if os.path.exists(self.text_chunks_path):
            print(f"Loading text chunks from {self.text_chunks_path}...")
            self.text_chunks_df = pd.read_pickle(self.text_chunks_path)
            print(f"Loaded {len(self.text_chunks_df)} text chunks")

        # Load structured data
        if os.path.exists(self.structured_data_path):
            print(f"Loading structured data from {self.structured_data_path}...")
            with open(self.structured_data_path, 'rb') as f:
                self.structured_data = pickle.load(f)
            print(f"Loaded structured data with {len(self.structured_data)} files")

            # Create descriptions for structured data
            self.create_structured_data_descriptions()

    def create_structured_data_descriptions(self):
        """Create searchable descriptions for structured data."""
        if not self.structured_data:
            return

        descriptions = []
        for item in self.structured_data:
            df = item['data']
            filename = item['filename']

            # Create a description of the dataset
            description = f"Dataset: {filename}\n"
            description += f"Contains {len(df)} rows and {len(df.columns)} columns.\n"
            description += f"Columns: {', '.join(df.columns.tolist())}\n"

            # Add sample data (first few rows)
            description += "Sample data:\n"
            sample = df.head(3).to_string()
            description += sample

            descriptions.append({
                'source_file': filename,
                'text': description,
                'type': 'structured'
            })

            # Also add column-specific descriptions
            for column in df.columns:
                col_desc = f"Column '{column}' in dataset {filename}.\n"
                try:
                    # Add statistical information if numeric
                    if pd.api.types.is_numeric_dtype(df[column]):
                        col_desc += f"Min: {df[column].min()}, Max: {df[column].max()}, Mean: {df[column].mean():.2f}\n"
                    # Add unique values if categorical with few values
                    elif df[column].nunique() < 10:
                        col_desc += f"Values: {', '.join(map(str, df[column].unique()[:10]))}\n"
                    # Add value counts for high-frequency items
                    value_counts = df[column].value_counts().nlargest(5).to_dict()
                    col_desc += "Most common values: " + ", ".join([f"{k}: {v}" for k, v in value_counts.items()])
                except:
                    # Skip if there's an error calculating stats
                    pass

                descriptions.append({
                    'source_file': filename,
                    'text': col_desc,
                    'type': 'structured_column',
                    'column': column
                })

        self.structured_data_descriptions = pd.DataFrame(descriptions)

    def create_embeddings(self, save_text_path="text_embeddings.pkl", save_structured_path="structured_embeddings.pkl"):
        """Create embeddings for all content."""
        # Create text embeddings
        if self.text_chunks_df is not None:
            print(f"Creating text embeddings using {self.embedding_model_name}...")
            texts = self.text_chunks_df['text'].tolist()

            # Embed in batches
            batch_size = 64
            embeddings = []

            for i in tqdm(range(0, len(texts), batch_size), desc="Creating text embeddings"):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                embeddings.append(batch_embeddings)

            self.text_embeddings = np.vstack(embeddings)

            # Save embeddings
            if save_text_path:
                with open(save_text_path, 'wb') as f:
                    pickle.dump(self.text_embeddings, f)
                print(f"Saved text embeddings to {save_text_path}")

        # Create structured data embeddings
        if self.structured_data_descriptions is not None:
            print(f"Creating structured data embeddings using {self.embedding_model_name}...")
            descriptions = self.structured_data_descriptions['text'].tolist()

            # Embed in batches
            batch_size = 64
            embeddings = []

            for i in tqdm(range(0, len(descriptions), batch_size), desc="Creating structured data embeddings"):
                batch_texts = descriptions[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                embeddings.append(batch_embeddings)

            self.structured_data_embeddings = np.vstack(embeddings)

            # Save embeddings
            if save_structured_path:
                with open(save_structured_path, 'wb') as f:
                    pickle.dump(self.structured_data_embeddings, f)
                print(f"Saved structured data embeddings to {save_structured_path}")

    def load_embeddings(self, text_embeddings_path="text_embeddings.pkl",
                        structured_embeddings_path="structured_embeddings.pkl"):
        """Load pre-computed embeddings."""
        # Load text embeddings
        if os.path.exists(text_embeddings_path):
            print(f"Loading text embeddings from {text_embeddings_path}...")
            with open(text_embeddings_path, 'rb') as f:
                self.text_embeddings = pickle.load(f)
            print(f"Loaded text embeddings with shape {self.text_embeddings.shape}")

        # Load structured data embeddings
        if os.path.exists(structured_embeddings_path) and self.structured_data_descriptions is not None:
            print(f"Loading structured data embeddings from {structured_embeddings_path}...")
            with open(structured_embeddings_path, 'rb') as f:
                self.structured_data_embeddings = pickle.load(f)
            print(f"Loaded structured data embeddings with shape {self.structured_data_embeddings.shape}")

    def build_indices(self):
        """Build FAISS indices for fast similarity search."""
        # Build text index
        if self.text_embeddings is not None:
            print("Building text FAISS index...")
            dimension = self.text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(dimension)
            self.text_index.add(self.text_embeddings.astype('float32'))
            print(f"Built text index with {self.text_index.ntotal} vectors")

        # Build structured data index
        if self.structured_data_embeddings is not None:
            print("Building structured data FAISS index...")
            dimension = self.structured_data_embeddings.shape[1]
            self.structured_data_index = faiss.IndexFlatL2(dimension)
            self.structured_data_index.add(self.structured_data_embeddings.astype('float32'))
            print(f"Built structured data index with {self.structured_data_index.ntotal} vectors")

    def classify_query(self, query):
        """Classify whether the query is more likely to be for text or structured data."""
        # Keywords that suggest structured data queries
        structured_keywords = [
            'how many', 'average', 'median', 'statistics', 'price range', 'data',
            'dataset', 'spreadsheet', 'chart', 'graph', 'table', 'column', 'row',
            'excel', 'csv', 'percentage', 'trend', 'distribution', 'count'
        ]

        # Simple rule-based classification
        if any(keyword in query.lower() for keyword in structured_keywords):
            return "structured"

        # More complex classification could be implemented here
        return "text"

    def search(self, query, k=5, query_type=None):
        """
        Search for content most similar to the query.

        Args:
            query: The search query
            k: Number of results to return
            query_type: Force search in "text" or "structured" data, or "both"

        Returns:
            List of dictionaries with content and metadata
        """
        # Auto-classify query if not specified
        if query_type is None:
            query_type = self.classify_query(query)

        # Embed the query
        query_embedding = self.embedding_model.encode([query])

        results = []

        # Search text index
        if query_type in ["text", "both"] and self.text_index is not None:
            distances, indices = self.text_index.search(query_embedding.astype('float32'), k)

            for i, idx in enumerate(indices[0]):
                if idx < len(self.text_chunks_df):  # Ensure index is valid
                    chunk = self.text_chunks_df.iloc[idx]
                    results.append({
                        'chunk_id': chunk['chunk_id'],
                        'source_file': chunk['source_file'],
                        'text': chunk['text'],
                        'distance': distances[0][i],
                        'type': 'text'
                    })

        # Search structured data index
        if query_type in ["structured", "both"] and self.structured_data_index is not None:
            # Adjust k to get enough results when combining
            s_k = k if query_type == "structured" else k // 2

            distances, indices = self.structured_data_index.search(query_embedding.astype('float32'), s_k)

            for i, idx in enumerate(indices[0]):
                if idx < len(self.structured_data_descriptions):  # Ensure index is valid
                    item = self.structured_data_descriptions.iloc[idx]
                    results.append({
                        'source_file': item['source_file'],
                        'text': item['text'],
                        'distance': distances[0][i],
                        'type': item['type']
                    })

        # Sort by distance
        results.sort(key=lambda x: x['distance'])

        # Limit to k results
        return results[:k]

    def answer_question(self, question, k=5):
        """
        Answer a question using RAG.

        Args:
            question: The question to answer
            k: Number of chunks to retrieve

        Returns:
            Dictionary with retrieved context and metadata
        """
        # Determine if this is a structured or unstructured data question
        query_type = self.classify_query(question)

        # Retrieve relevant content
        relevant_content = self.search(question, k=k, query_type=query_type)

        # Combine context
        context = "\n\n".join([f"From {item['source_file']}:\n{item['text']}"
                               for item in relevant_content])

        return {
            'question': question,
            'retrieved_content': relevant_content,
            'context': context,
            'query_type': query_type
        }

    def extract_structured_data(self, filename, column=None, filters=None):
        """
        Extract specific data from structured datasets.

        Args:
            filename: Name of the dataset file
            column: Specific column to extract
            filters: Dictionary of column:value pairs to filter the data

        Returns:
            DataFrame or Series with the extracted data
        """
        if not self.structured_data:
            return None

        # Find the dataset
        dataset = None
        for item in self.structured_data:
            if item['filename'] == filename:
                dataset = item['data']
                break

        if dataset is None:
            return None

        # Apply filters
        if filters:
            for col, value in filters.items():
                if col in dataset.columns:
                    dataset = dataset[dataset[col] == value]

        # Extract specific column
        if column and column in dataset.columns:
            return dataset[column]

        return dataset