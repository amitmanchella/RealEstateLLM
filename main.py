import os
import argparse
import PyPDF2
import pandas as pd
from tqdm import tqdm
import re
import unicodedata
import pickle
import numpy as np
from data_processors import DataProcessor, StructuredDataProcessor, DocProcessor
from enhanced_rag import EnhancedRealEstateRAG
from real_estate_chatbot import RealEstateChatbot
from fine_tuning import FineTuningDataGenerator


# PDF Processing Functions from the notebook
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        return text


def process_pdfs(directory):
    """Process all PDFs in the given directory."""
    results = []
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        file_path = os.path.join(directory, pdf_file)
        try:
            text = extract_text_from_pdf(file_path)
            results.append({
                'filename': pdf_file,
                'text': text,
                'size': os.path.getsize(file_path)
            })
            print(f"Successfully processed {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df


def clean_text(text):
    """Clean and preprocess text for NLP tasks."""
    # Replace common non-ASCII characters
    text = text.replace('–', '-').replace('—', '-').replace(''', "'").replace(''', "'")
    text = text.replace('"', '"').replace('"', '"').replace('…', '...')

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up page numbers and headers/footers (common in PDFs)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text)  # Page indicators
    text = re.sub(r'\n\s*[Pp]age\s*\n', '\n', text)  # More page indicators

    # Remove common PDF artifacts
    text = re.sub(r'(?:\n|^)\s*([ivx]+|[IVX]+)\s*(?:\n|$)', '\n', text)  # Roman numerals
    text = re.sub(r'(?:\n|^)\s*(\d+\.\d+\.\d+)\s*(?:\n|$)', '\n', text)  # Document section numbers

    # Remove document reference numbers
    text = re.sub(r'\b([A-Z]{2,}\d+)\b', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    return text.strip()


def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks that preserve semantic units."""
    if len(text) <= chunk_size:
        return [text]

    # Look for section headers to use as better breaking points
    section_pattern = re.compile(r'(?:\n|^)\s*([A-Z][A-Za-z\s]{2,}:|\d+\.\s+[A-Z]|\d+\s+[A-Z]|[A-Z][A-Z\s]+\n)')
    section_matches = list(section_pattern.finditer(text))
    
    chunks = []
    current_pos = 0
    
    # Process text section by section when possible
    for i in range(len(section_matches)):
        section_start = section_matches[i].start()
        
        # Determine section end (next section or end of text)
        if i < len(section_matches) - 1:
            section_end = section_matches[i+1].start()
        else:
            section_end = len(text)
        
        # If section is too large, apply internal chunking
        section_text = text[section_start:section_end]
        if len(section_text) > chunk_size * 1.5:
            # Split into paragraphs first
            paragraphs = [p for p in section_text.split('\n\n') if p.strip()]
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    # If current chunk has content, add it
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if current_chunk else ""
                    current_chunk = overlap_text + para + "\n\n"
            
            # Add the last chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            # Add the entire section as one chunk
            chunks.append(section_text.strip())
    
    # If no chunks were created (no good section breaks found)
    if not chunks:
        # Fall back to simpler chunking method
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Find a good breaking point
            if end < len(text):
                # Look for paragraph breaks
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1:
                    end = paragraph_break
                else:
                    # Look for sentence breaks
                    for marker in ['. ', '! ', '? ']:
                        sentence_break = text.rfind(marker, start, end)
                        if sentence_break != -1:
                            end = sentence_break + 2
                            break
            
            chunks.append(text[start:end].strip())
            start = max(start + chunk_size - overlap, end)
    
    return chunks


def process_text_data(raw_df, output_path=None, chunk_size=1500):
    """Clean, preprocess and chunk text data from PDFs."""
    print("Cleaning text data...")
    # Use tqdm to show progress
    # tqdm.pandas is used in the notebook, but for simplicity we'll use a regular loop here
    cleaned_texts = []
    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Cleaning texts"):
        cleaned_texts.append(clean_text(row['text']))
    raw_df['cleaned_text'] = cleaned_texts

    # Chunk texts
    print("Chunking documents into smaller pieces...")
    all_chunks = []

    for i, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Chunking documents"):
        try:
            # Skip extremely large docs or process them differently
            if len(row['cleaned_text']) > 1_000_000:  # 1 million chars
                print(f"⚠️ Large document detected: {row['filename']} ({len(row['cleaned_text'])} chars)")
                # Process large documents in a simpler way (just divide by size)
                simple_chunks = [row['cleaned_text'][j:j + chunk_size]
                                 for j in range(0, len(row['cleaned_text']), chunk_size)]
                for j, chunk in enumerate(simple_chunks):
                    all_chunks.append({
                        'source_file': row['filename'],
                        'chunk_id': f"{row['filename']}_simple_{j}",
                        'text': chunk.strip()
                    })
                continue

            # Regular chunking for normal sized documents
            chunks = chunk_text(row['cleaned_text'], chunk_size=chunk_size)
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    'source_file': row['filename'],
                    'chunk_id': f"{row['filename']}_{j}",
                    'text': chunk
                })
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    chunks_df = pd.DataFrame(all_chunks)
    print(f"Created {len(chunks_df)} chunks from {len(raw_df)} documents")

    # Save the processed data
    if output_path:
        chunks_df.to_pickle(output_path)
        print(f"Saved processed chunks to {output_path}")

    return chunks_df


def process_all_data(data_dir="data", force_reprocess=False):
    """Process all data sources and generate necessary files."""
    print("Starting data processing...")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist. Creating it...")
        os.makedirs(data_dir)
        print(f"Please add your data files to the '{data_dir}' directory and run again.")
        return

    # Check files in data directory
    files = os.listdir(data_dir)
    if not files:
        print(f"No files found in '{data_dir}' directory. Please add your data files and run again.")
        return

    print(f"Found {len(files)} files in '{data_dir}' directory:")
    for file in files[:10]:  # Show first 10 files
        print(f"- {file}")
    if len(files) > 10:
        print(f"... and {len(files) - 10} more files")

    # Check if processed files already exist
    text_chunks_exists = os.path.exists("processed_chunks.pkl")
    structured_data_exists = os.path.exists("processed_structured_data.pkl")
    doc_data_exists = os.path.exists("processed_doc_data.pkl")

    if text_chunks_exists and structured_data_exists and doc_data_exists and not force_reprocess:
        print("Processed data files already exist:")
        if text_chunks_exists:
            print("- processed_chunks.pkl (Found)")
        if structured_data_exists:
            print("- processed_structured_data.pkl (Found)")
        if doc_data_exists:
            print("- processed_doc_data.pkl (Found)")
        print("Use --force flag to reprocess.")
        return

    # Process PDF files (using code from the notebook)
    if not text_chunks_exists or force_reprocess:
        print("\nProcessing PDF files...")
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        if not pdf_files:
            print("No PDF files found. Skipping PDF processing.")
        else:
            print(f"Found {len(pdf_files)} PDF files to process.")
            # Process PDFs and create chunks
            results_df = process_pdfs(data_dir)
            results_df.to_pickle("extracted_pdf_texts.pkl")
            print(f"Processed {len(results_df)} PDF files and saved to extracted_pdf_texts.pkl")

            # Process the extracted texts into cleaned chunks
            processed_df = process_text_data(results_df, "processed_chunks.pkl")
            print(f"Processed chunks statistics:")
            print(f"- Total chunks: {len(processed_df)}")
            print(f"- Average chunk length: {processed_df['text'].str.len().mean():.1f} characters")
    else:
        print("\nSkipping PDF processing (processed_chunks.pkl already exists).")

    # Process structured data
    if not structured_data_exists or force_reprocess:
        print("\nProcessing structured data (CSV/Excel files)...")
        structured_processor = StructuredDataProcessor(data_dir)
        structured_data = structured_processor.process_all()
        if structured_data:
            structured_processor.save_processed_data(structured_data, "processed_structured_data.pkl")
            print(f"Processed {len(structured_data)} structured data files")
        else:
            print("No structured data files found or processed.")
    else:
        print("\nSkipping structured data processing (processed_structured_data.pkl already exists).")

    # Process Word documents
    if not doc_data_exists or force_reprocess:
        print("\nProcessing Word documents...")
        doc_processor = DocProcessor(data_dir)
        doc_data = doc_processor.process_all()
        if doc_data:
            doc_processor.save_processed_data(doc_data, "processed_doc_data.pkl")
            print(f"Processed {len(doc_data)} Word documents")
        else:
            print("No Word documents found or processed.")
    else:
        print("\nSkipping Word document processing (processed_doc_data.pkl already exists).")

    print("\nData processing complete!")


def create_embeddings(force_recreate=False):
    """Create embeddings for all data sources."""
    print("Starting embedding creation...")

    # Check if necessary files exist
    text_chunks_exists = os.path.exists("processed_chunks.pkl")
    structured_data_exists = os.path.exists("processed_structured_data.pkl")

    if not text_chunks_exists and not structured_data_exists:
        print("No processed data files found. Please run --process first.")
        return

    text_embeddings_exists = os.path.exists("text_embeddings.pkl")
    structured_embeddings_exists = os.path.exists("structured_embeddings.pkl")

    if text_embeddings_exists and structured_embeddings_exists and not force_recreate:
        print("Embedding files already exist:")
        if text_embeddings_exists:
            print("- text_embeddings.pkl (Found)")
        if structured_embeddings_exists:
            print("- structured_embeddings.pkl (Found)")
        print("Use --force flag to recreate.")
        return

    # Initialize RAG system
    print("Initializing RAG system...")
    rag = EnhancedRealEstateRAG()

    # Create embeddings
    print("Creating embeddings...")
    rag.create_embeddings()
    print("Embedding creation complete!")


def prepare_fine_tuning_data(num_pairs=1000):
    """Prepare data for fine-tuning experiments."""
    print(f"Generating fine-tuning data with {num_pairs} QA pairs...")

    # Check if necessary files exist
    text_chunks_exists = os.path.exists("processed_chunks.pkl")

    if not text_chunks_exists:
        print("No processed text chunks found. Please run --process first.")
        return

    # Initialize data generator
    generator = FineTuningDataGenerator()

    # Generate QA pairs
    qa_pairs = generator.generate_qa_pairs(num_pairs=num_pairs)

    # Format for different fine-tuning approaches
    generator.generate_fine_tuning_formats(qa_pairs)

    print("Fine-tuning data preparation complete!")


def start_chatbot():
    """Initialize and start the chatbot."""
    print("Initializing Real Estate Chatbot...")

    # Check if necessary files exist
    text_embeddings_exists = os.path.exists("text_embeddings.pkl")

    if not text_embeddings_exists:
        print("Required text embeddings file not found. Please run --embeddings first.")
        return

    # Initialize RAG system
    rag = EnhancedRealEstateRAG()
    rag.load_embeddings()
    rag.build_indices()

    # Initialize chatbot
    chatbot = RealEstateChatbot(rag)

    # Start interactive chat
    chatbot.chat()


def main():
    parser = argparse.ArgumentParser(description='RealEstateLLM - AI-powered real estate chatbot')

    parser.add_argument('--process', action='store_true',
                        help='Process all data sources')
    parser.add_argument('--embeddings', action='store_true',
                        help='Create embeddings for all data')
    parser.add_argument('--finetune', action='store_true',
                        help='Prepare data for fine-tuning')
    parser.add_argument('--chat', action='store_true',
                        help='Start the chatbot')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of existing files')
    parser.add_argument('--qa-pairs', type=int, default=1000,
                        help='Number of QA pairs to generate for fine-tuning')

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Process data if requested
    if args.process:
        process_all_data(force_reprocess=args.force)

    # Create embeddings if requested
    if args.embeddings:
        create_embeddings(force_recreate=args.force)

    # Prepare fine-tuning data if requested
    if args.finetune:
        prepare_fine_tuning_data(num_pairs=args.qa_pairs)

    # Start chatbot if requested
    if args.chat:
        start_chatbot()


if __name__ == "__main__":
    main()