"""
PDF to Text Converter
---------------------
This script extracts text from PDF files in the 'rules' directory
and saves it as text files in the 'text' directory.
"""

import os
import pypdf
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def main():
    # Create directories if they don't exist
    rules_dir = Path("rules")
    text_dir = Path("text")
    
    if not text_dir.exists():
        text_dir.mkdir(exist_ok=True)
        print(f"Created directory: {text_dir}")
    
    # Get all PDF files from the rules directory
    pdf_files = list(rules_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the 'rules' directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files.")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        
        # Save text to file
        text_file = text_dir / f"{pdf_file.stem}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved text to: {text_file}")
    
    print("All PDFs processed successfully.")

if __name__ == "__main__":
    main()
