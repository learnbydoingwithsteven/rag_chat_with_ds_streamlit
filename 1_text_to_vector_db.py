"""
Text to Vector Database Converter
--------------------------------
This script reads text files from the 'text' directory,
splits them into chunks, generates embeddings,
and stores them in a SQLite database for vector search.
"""

import os
os.environ['USE_TF'] = '0'
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
DB_PATH = "bilanci_vectors.db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_database(force_reset=False):
    """Create the SQLite database with necessary tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if tables already exist
    existing_tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    existing_tables = [table[0] for table in existing_tables]
    
    # If force_reset is True or paragraphs table exists but doesn't have the right columns, drop all tables
    if force_reset or 'paragraphs' in existing_tables:
        print("Recreating database tables...")
        # Drop existing tables
        if 'paragraphs' in existing_tables:
            cursor.execute("DROP TABLE IF EXISTS paragraphs")
        if 'metadata' in existing_tables:
            cursor.execute("DROP TABLE IF EXISTS metadata")
    
    # Create paragraphs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paragraphs (
        id INTEGER PRIMARY KEY,
        content TEXT,
        source TEXT,
        embedding BLOB
    )
    ''')
    
    # Create metadata table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")

def split_text_into_chunks(text, source):
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return [(chunk, source) for chunk in chunks]

def get_embeddings(chunks, model):
    """Generate embeddings for text chunks using SentenceTransformer"""
    texts = [chunk[0] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings

def store_in_database(chunks, embeddings):
    """Store text chunks and their embeddings in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Insert each chunk with its embedding
    for i, ((chunk, source), embedding) in enumerate(zip(chunks, embeddings)):
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO paragraphs (content, source, embedding) VALUES (?, ?, ?)",
            (chunk, source, embedding_blob)
        )
    
    conn.commit()
    conn.close()
    print(f"Stored {len(chunks)} chunks in the database")

def process_text_files():
    """Process all text files in the 'text' directory"""
    # Get the directory where the script is located
    script_dir = Path(__file__).parent.absolute()
    text_dir = script_dir / "text"
    
    print(f"Looking for text files in: {text_dir}")
    
    if not text_dir.exists():
        print("Text directory not found. Please run 0_pdf_to_text.py first.")
        return
    
    # Initialize the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Create the database (force reset to ensure correct schema)
    create_database(force_reset=True)
    
    # Get all text files
    text_files = list(text_dir.glob("*.txt"))
    
    if not text_files:
        print("No text files found in the 'text' directory.")
        return
    
    print(f"Found {len(text_files)} text files.")
    
    # Process each text file
    all_chunks = []
    
    for text_file in text_files:
        print(f"Processing: {text_file.name}")
        
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks
        chunks = split_text_into_chunks(text, text_file.stem)
        all_chunks.extend(chunks)
        
        print(f"  - Split into {len(chunks)} chunks")
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = get_embeddings(all_chunks, model)
    
    # Store chunks and embeddings in database
    store_in_database(all_chunks, embeddings)
    
    print("All text files processed and stored in the database.")

if __name__ == "__main__":
    process_text_files()
