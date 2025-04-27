"""
Bilanci Armonizzati Chatbot
--------------------------
This Streamlit application provides a RAG-based chatbot interface
to interact with and query information from the Bilanci Armonizzati documents.
Supports both English and Italian languages.
"""

import os
os.environ['USE_TF'] = '0'
import tempfile
import streamlit as st
import sqlite3
import numpy as np
import json
import base64
import time
import pandas as pd
import io
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import groq
import PyPDF2
import requests
import textwrap
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import UMAP, but make it optional
try:
    import umap.umap_ as umap_module
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from gtts import gTTS

# Load environment variables
load_dotenv()

# Constants
# Use absolute paths to avoid directory issues
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bilanci_vectors.db")
PDF_DIR = os.path.join(BASE_DIR, "rules")

# BDAP CKAN API constants
CKAN_ROOT = "https://bdap-opendata.rgs.mef.gov.it/SpodCkanApi/api/3/action"
BDAP_ALT_ROOT = "https://bdap-opendata.rgs.mef.gov.it/SpodCkanApi/api/1/rest/dataset"

# BDAP API Helper Functions
def ckan_api(action: str, params: dict | None = None, max_retries=3, timeout=20):
    """Call CKAN Action API with retry mechanism"""
    url = f"{CKAN_ROOT}/{action}"
    
    for attempt in range(max_retries):
        try:
            # Use a shorter timeout and implement retry logic
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            out = resp.json()
            
            if not out.get("success", False):
                raise RuntimeError(out.get("error"))
                
            return out["result"]
        
        except (requests.ConnectionError, requests.Timeout) as e:
            # Handle connection issues with clearer messages
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt+1}/{max_retries} failed. Retrying in 2 seconds...")
                time.sleep(2)  # Wait before retrying
            else:
                raise RuntimeError(f"Failed to connect to BDAP API after {max_retries} attempts. Server may be temporarily unavailable.") from e
        
        except Exception as e:
            # Handle other errors
            raise RuntimeError(f"BDAP API error: {str(e)}") from e

def find_csv_resource(year: int, regione: str, data_type: str) -> dict:
    """Find CSV resource for a specific year and region"""
    q = f'"{regione}" AND "{data_type} Enti Locali" AND {year}'
    res = ckan_api("package_search", {"q": q, "rows": 1})
    if res["count"] == 0:
        raise LookupError(f"No dataset for {regione} {year} {data_type}")
    pkg = res["results"][0]
    # Pick first CSV resource
    rsrc = next((r for r in pkg["resources"] if r["format"].lower() == "csv"), None)
    if not rsrc:
        raise LookupError(f"No CSV resource found for {regione} {year} {data_type}")
    return {"year": year, "url": rsrc["url"], "name": rsrc["name"]}
    
# Alternative BDAP API Functions
def bdap_alt_api(endpoint="", max_retries=3, timeout=15):
    """Call alternative BDAP API with retry mechanism"""
    url = f"{BDAP_ALT_ROOT}/{endpoint}".rstrip('/')
    
    for attempt in range(max_retries):
        try:
            # Use a moderate timeout and implement retry logic
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        
        except (requests.ConnectionError, requests.Timeout) as e:
            # Handle connection issues with clearer messages
            if attempt < max_retries - 1:
                # Use st.info instead of st.warning to avoid cluttering the UI
                if "st" in globals():
                    st.info(f"Connection attempt {attempt+1}/{max_retries} failed. Retrying...")
                time.sleep(2)  # Wait before retrying
            else:
                raise RuntimeError(f"Failed to connect to alternative BDAP API after {max_retries} attempts. Server may be temporarily unavailable.") from e
        
        except Exception as e:
            # Handle other errors
            raise RuntimeError(f"Alternative BDAP API error: {str(e)}")

def get_alt_datasets(filter_keyword=None):
    """Get list of available datasets from alternative BDAP API"""
    datasets = bdap_alt_api()
    if filter_keyword:
        return [d for d in datasets if filter_keyword.lower() in d.lower()]
    return datasets

def get_alt_dataset_details(dataset_id):
    """Get details for a specific dataset"""
    return bdap_alt_api(dataset_id)

# Set page configuration (must be called as the first Streamlit command)
st.set_page_config(
    page_title="Harmonized Financial Statements Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available LLM models - Groq models
GROQ_MODELS = {
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Llama 3 8B": "llama3-8b-8192",
    "Llama 3 70B": "llama3-70b-8192",
    "Gemma2 9B": "gemma2-9b-it",
}

# Function to get available Ollama models
def get_available_ollama_models():
    """Get a list of available Ollama models by calling 'ollama list'"""
    try:
        import subprocess
        import re
        
        # Run 'ollama list' to get models in standard format
        result = subprocess.run(["ollama", "list"], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        if result.returncode != 0:
            st.warning(f"Error running 'ollama list': {result.stderr}")
            return {"No models found (Ollama error)": ""}
        
        models = {}
        # Parse the standard output format
        # Example format: NAME                    ID              SIZE    MODIFIED
        # We're looking for the first column (NAME)
        
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Only header or empty
            return {"No models found (install models with 'ollama pull')": ""}
            
        # Skip header line
        for line in lines[1:]:
            if line.strip():
                # Split the line by whitespace and get the first part (NAME)
                parts = re.split(r'\s{2,}', line.strip())
                if parts and len(parts) >= 1:
                    model_name = parts[0].strip()
                    if model_name:
                        # For display, clean up the model name
                        display_name = model_name
                        if ":" in display_name:
                            display_name = display_name.split(":")[0]  # Remove tags
                        models[display_name] = model_name
        
        # If no models were found
        if not models:
            return {"No models found (install models with 'ollama pull')": ""}
            
        return models
    except Exception as e:
        st.warning(f"Error getting Ollama models: {str(e)}")
        return {"Error getting models": ""}

# UI Text dictionary for multilingual support
UI_TEXT = {
    "English": {
        "page_title": "Harmonized Financial Statements Assistant",
        "header": "Configuration",
        "select_model": "Select LLM model:",
        "api_key": "Groq API Key:",
        "api_source": "API Source:",
        "groq": "Groq API",
        "ollama": "Local Ollama",
        "primary_api": "Local API (Local Budgets)",
        "alternative_api": "Complete API (BDAP Datasets)",
        "db_found": "✅ Vector database found",
        "db_not_found": "❌ Vector database not found",
        "db_instructions": "Run `0_pdf_to_text.py` and `1_text_to_vector_db.py` scripts first",
        "docs_available": "Available Documents",
        "no_docs": "No PDF documents found in the 'rules' folder",
        "enable_web": "Enable web search",
        "enable_tts": "Enable text-to-speech",
        "show_thinking": "Show reasoning process",
        "main_title": "Harmonized Financial Statements Assistant 📊",
        "description": "This assistant helps you understand and navigate documentation on harmonized financial statements. You can ask questions about requirements, controls, and procedures related to harmonized financial statements.",
        "sample_title": "Sample Questions",
        "sample_click": "Click on a question to try it:",
        "chat_placeholder": "Ask a question about harmonized financial statements...",
        "processing": "Processing response...",
        "reasoning": "Reasoning Process",
        "context": "Relevant Context",
        "no_context": "No relevant context found in the database.",
        "sources": "Consulted Sources",
        "web_results": "Web Results",
        "upload_pdf": "Upload PDF",
        "upload_instructions": "Upload a PDF file to include in the knowledge base:",
        "upload_success": "✅ PDF uploaded successfully",
        "upload_error": "❌ Error uploading PDF",
        "download_chat": "Download Chat History",
        "feedback_title": "Was this response helpful?",
        "feedback_yes": "Yes",
        "feedback_no": "No",
        "feedback_thanks": "Thank you for your feedback!",
        "continue_questions": "Continue with more sample questions",
        "ask_more": "Ask another question",
        "restart_chat": "Restart chat",
        "view_pdf": "View PDF",
        "rag_settings": "RAG Settings",
        "chunk_size": "Chunk Size",
        "chunk_overlap": "Chunk Overlap",
        "similarity_top_k": "Number of similar documents",
        "doc_insights": "Document Insights",
        "doc_stats": "Document Statistics",
        "tab_chat": "RAG Chat",
        "tab_docs": "Documents",
        "tab_vis": "Vector Visualizations",
        "tab_data": "Financial Data",
        "tab_settings": "RAG Settings(new pdf)",
        "vis_title": "Embeddings Visualization",
        "vis_description": "Visualize document embeddings in 2D space. Each point represents a text chunk.",
        "vis_dim_reduction": "Dimensionality Reduction Method:",
        "pdf_chunks": "PDF Chunks",
        "text_chunks": "Text File Chunks",
        
        "data_header": "Italian Local Government Finance Data",
        "data_description": "Explore financial data from Italian local governments using BDAP open data.",
        "region_select": "Select Region",
        "data_type": "Data Type",
        "years_range": "Year Range",
        "fetch_data": "Fetch Data",
        "trend_title": "Time Trend Analysis",
        "structure_title": "Structure Analysis",
        "top_n": "Show Top Categories",
        "no_data": "No data available. Please fetch data first."
    },
    "Italiano": {
        "page_title": "Assistente Bilanci Armonizzati",
        "header": "Configurazione",
        "select_model": "Seleziona modello LLM:",
        "api_key": "Groq API Key:",
        "api_source": "Fonte API:",
        "groq": "API Groq",
        "ollama": "Ollama Locale",
        "primary_api": "API Locale (Bilanci Enti Locali)",
        "alternative_api": "API Completa (Dataset BDAP)",
        "db_found": "✅ Database vettoriale trovato",
        "db_not_found": "❌ Database vettoriale non trovato",
        "db_instructions": "Esegui prima gli script `0_pdf_to_text.py` e `1_text_to_vector_db.py`",
        "docs_available": "Documenti disponibili",
        "no_docs": "Nessun documento PDF trovato nella cartella 'rules'",
        "enable_web": "Abilita ricerca web",
        "enable_tts": "Abilita sintesi vocale",
        "show_thinking": "Mostra processo di ragionamento",
        "main_title": "Assistente Bilanci Armonizzati 📊",
        "description": "Questo assistente ti aiuta a comprendere e navigare la documentazione sui bilanci armonizzati. Puoi fare domande sui requisiti, controlli, e procedure relative ai bilanci armonizzati.",
        "sample_title": "Domande di esempio",
        "sample_click": "Clicca su una domanda per provarla:",
        "chat_placeholder": "Fai una domanda sui bilanci armonizzati...",
        "processing": "Elaborando la risposta...",
        "reasoning": "Processo di ragionamento",
        "context": "Contesto rilevante",
        "no_context": "Nessun contesto rilevante trovato nel database.",
        "sources": "Fonti consultate",
        "web_results": "Risultati web",
        "upload_pdf": "Carica PDF",
        "upload_instructions": "Carica un file PDF da includere nella base di conoscenza:",
        "upload_success": "✅ PDF caricato con successo",
        "upload_error": "❌ Errore nel caricamento del PDF",
        "download_chat": "Scarica Cronologia Chat",
        "feedback_title": "Questa risposta è stata utile?",
        "feedback_yes": "Sì",
        "feedback_no": "No",
        "feedback_thanks": "Grazie per il tuo feedback!",
        "continue_questions": "Continua con altre domande di esempio",
        "ask_more": "Fai un'altra domanda",
        "restart_chat": "Ricomincia chat",
        "view_pdf": "Visualizza PDF",
        "rag_settings": "Impostazioni RAG",
        "chunk_size": "Dimensione Chunk",
        "chunk_overlap": "Sovrapposizione Chunk",
        "similarity_top_k": "Numero di documenti simili",
        "doc_insights": "Analisi Documenti",
        "doc_stats": "Statistiche Documenti",
        "tab_chat": "Chat RAG",
        "tab_docs": "Documenti PDF",
        "tab_vis": "Visualizzazioni Vettoriali",
        "tab_data": "Dati Finanziari",
        "tab_settings": "Impostazioni RAG(new pdf)",
        "vis_title": "Visualizzazione Embedding",
        "vis_description": "Visualizza gli embedding dei documenti in spazio 2D. Ogni punto rappresenta un frammento di testo.",
        "vis_dim_reduction": "Metodo di Riduzione Dimensionale:",
        "pdf_chunks": "Frammenti PDF",
        "text_chunks": "Frammenti di File di Testo",
        
        "data_header": "Dati Finanziari degli Enti Locali Italiani",
        "data_description": "Esplora i dati finanziari degli enti locali italiani utilizzando i dati aperti BDAP.",
        "region_select": "Seleziona Regione",
        "data_type": "Tipo di Dati",
        "years_range": "Intervallo Anni",
        "fetch_data": "Scarica Dati",
        "trend_title": "Analisi delle Tendenze",
        "structure_title": "Analisi della Struttura",
        "top_n": "Mostra Principali Categorie",
        "no_data": "Nessun dato disponibile. Per favore scarica i dati prima."  
    }
}

# Sample questions dictionary
SAMPLE_QUESTIONS = {
    "English": [
        "What are harmonized financial statements?",
        "What are the requirements for compiling Xbrl instances?",
        "How do I register with BDAP?",
        "What controls are performed on harmonized financial statements?",
        "Who must transmit harmonized accounting documents?",
        "What are the deadlines for submitting harmonized financial statements?",
        "How do I handle corrections after submission?",
        "What are the technical specifications for XBRL taxonomy?",
        "How to validate XBRL instances before submission?",
        "What are the consequences of late or incorrect submissions?"
    ],
    "Italiano": [
        "Cosa sono i bilanci armonizzati?",
        "Quali sono i requisiti per la compilazione delle istanze Xbrl?",
        "Come si registra al BDAP?",
        "Quali controlli vengono effettuati sui bilanci armonizzati?",
        "Chi deve trasmettere i documenti contabili armonizzati?",
        "Quali sono le scadenze per la presentazione dei bilanci armonizzati?",
        "Come gestire le correzioni dopo l'invio?",
        "Quali sono le specifiche tecniche per la tassonomia XBRL?",
        "Come validare le istanze XBRL prima dell'invio?",
        "Quali sono le conseguenze per invii tardivi o errati?"
    ]
}

# System prompts dictionary
SYSTEM_PROMPTS = {
    "English": """
    You are an expert assistant on Italian harmonized financial statements.
    Answer user questions based on the provided documents.
    If you don't know the answer, clearly state that instead of making up information.
    """,
    "Italiano": """
    Sei un assistente esperto in bilanci armonizzati italiani.
    Rispondi alle domande dell'utente basandoti sui documenti forniti.
    Se non conosci la risposta, dillo chiaramente invece di inventare informazioni.
    """
}

# Function to get embedding for text
def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    """Get embedding for text using SentenceTransformer"""
    model = SentenceTransformer(model_name)
    embedding = model.encode(text)
    return embedding

# Function to find similar paragraphs in the database
def find_similar_paragraphs(query_embedding, top_k=5):
    """Find paragraphs similar to the query embedding"""
    try:
        # Check if database exists
        if not Path(DB_PATH).exists():
            return []
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database structure
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        results = []
        
        # Handle different database structures
        if "paragraphs" in tables:  # Structure from actual 1_text_to_vector_db.py
            # Get paragraphs with embeddings
            cursor.execute("""
            SELECT id, content, source, embedding FROM paragraphs
            """)
            results = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
        
        elif "chunks" in tables:  # Original structure from older 1_text_to_vector_db.py
            # Get chunks with embeddings
            cursor.execute("""
            SELECT id, text, source, embedding FROM chunks
            """)
            results = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
        
        elif "documents" in tables and "embeddings" in tables:  # New structure
            # Retrieve all embeddings
            cursor.execute("""
            SELECT e.id, d.content, d.source, e.embedding
            FROM embeddings e
            JOIN documents d ON e.document_id = d.id
            """)
            results = cursor.fetchall()
        
        conn.close()
        
        if not results:
            st.warning("No embeddings found in the database.")
            return []
        
        # Calculate similarities
        similarities = []
        for id_, content, source, embedding_bytes in results:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append((id_, content, source, similarity))
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[3], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    except Exception as e:
        st.error(f"Error finding similar paragraphs: {str(e)}")
        return []

# Function to search the web
def search_web(query, max_results=3, language="it"):
    """Search the web using DuckDuckGo"""
    try:
        # Add language context to the query based on selected language
        search_query = query
        if language == "English":
            search_query = f"harmonized financial statements italy {query}"
        else:
            search_query = f"bilanci armonizzati {query}"
            
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))
        return results
    except Exception as e:
        st.error(f"Error searching the web: {e}")
        return []

# Function to get a conversation chain with Groq
# Not using LangChain for conversation chain anymore

# Function to generate audio from text
def generate_audio(text, language="it"):
    """Generate audio from text using gTTS"""
    try:
        # Map UI language to gTTS language
        tts_lang = "it" if language == "Italiano" else "en"
        
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, "rb") as f:
            audio_bytes = f.read()
        
        # Cleanup temp file
        os.unlink(tmp_file_path)
        
        # Encode to base64
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return base64_audio
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Function to get relevant context for a question
def get_relevant_context(question, top_k=5):
    """Get relevant context from the database for a question"""
    try:
        # Generate embedding for the question
        query_embedding = get_embedding(question)
        
        # Find similar paragraphs with customizable top_k
        results = find_similar_paragraphs(query_embedding, top_k=top_k)
        
        if not results:
            return "", []
        
        # Extract context and sources
        context = ""
        sources = []
        
        for result in results:
            # Extract content and source from result
            if result and len(result) >= 3:
                content = result[1] if result[1] else ""
                source = result[2] if result[2] else "Unknown source"
                
                # Add content to context
                context += f"{content}\n\n"
                
                # Add source to sources list if not already there
                if source and source not in sources:
                    sources.append(source)
        
        return context, sources
    except Exception as e:
        st.error(f"Error getting context: {str(e)}")
        return "", []

# Function to generate a response
def generate_response(question, context, sources, model_name, language="Italiano", api_source="groq"):
    """Generate a response using the context and model"""
    # Create a system message with context
    system_message = SYSTEM_PROMPTS[language]
    
    if context:
        if language == "English":
            system_message += "\n\nHere is some relevant information from the official documents:\n" + context
        else:
            system_message += "\n\nEcco alcune informazioni rilevanti dai documenti ufficiali:\n" + context
    
    try:
        if api_source == "groq":
            # Use Groq API for chat completion
            api_key = os.getenv("GROQ_API_KEY", "")
            
            if not api_key:
                return "No API key found for Groq. Please set the GROQ_API_KEY environment variable or use Ollama instead."
            
            client = groq.Client(api_key=api_key)
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                model=model_name,
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stream=False
            )
            
            # Extract the response
            response = chat_completion.choices[0].message.content
        else:  # api_source == "ollama"
            # Use local Ollama API directly
            try:
                # Prepare the request payload for Ollama API
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.7,
                    "stream": False
                }
                
                # Send request to Ollama API
                ollama_response = requests.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    timeout=60
                )
                
                # Check if the request was successful
                if ollama_response.status_code == 200:
                    # Extract the response
                    response_data = ollama_response.json()
                    response = response_data.get("message", {}).get("content", "")
                else:
                    return f"Error from Ollama API: Status code {ollama_response.status_code}. Make sure Ollama is running locally with 'ollama serve' command."
            except requests.RequestException as e:
                return f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running locally with 'ollama serve' command."
        
        # Add source citations
        if sources:
            if language == "English":
                source_text = "\n\n**Consulted Sources:**\n"
            else:
                source_text = "\n\n**Fonti consultate:**\n"
                
            for source in sources:
                source_text += f"- {source}\n"
            response += source_text
        
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        if language == "English":
            return f"Sorry, an error occurred while generating the response: {str(e)}"
        else:
            return f"Mi dispiace, si è verificato un errore durante la generazione della risposta: {str(e)}"

# Function to list available PDFs
def list_available_pdfs():
    """List all available PDFs in the rules directory"""
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        return []
    return [p.name for p in pdf_dir.glob("*.pdf")]

# Function to upload PDF files
def upload_pdf(file, language):
    """Upload a PDF file and add it to the knowledge base"""
    ui = UI_TEXT[language]
    try:
        # Create directory if it doesn't exist
        pdf_dir = Path(PDF_DIR)
        if not pdf_dir.exists():
            pdf_dir.mkdir(parents=True)
            
        # Save the uploaded file
        file_path = pdf_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
            
        # Extract text from PDF
        text_dir = Path("text")
        if not text_dir.exists():
            text_dir.mkdir(parents=True)
            
        text_path = text_dir / f"{file.name.rsplit('.', 1)[0]}.txt"
        
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + '\n'
                
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
            
        # Get RAG settings from session state for text processing
        chunk_size = 1000  # Default
        chunk_overlap = 200  # Default
        
        # Use session state values if available
        if "chunk_size" in st.session_state:
            chunk_size = st.session_state.chunk_size
        if "chunk_overlap" in st.session_state:
            chunk_overlap = st.session_state.chunk_overlap
        
        # Display notification about chunk settings being used
        st.info(f"Processing PDF with chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        # Process text for vector database with custom chunk settings
        process_text_for_db(text_path, language, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        st.success(ui["upload_success"])
        return True
    except Exception as e:
        st.error(f"{ui['upload_error']}: {str(e)}")
        return False

# Function to process text for vector database
def process_text_for_db(text_path, language, chunk_size=1000, chunk_overlap=200):
    """Process a text file and add it to the vector database"""
    try:
        # Read text file
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database structure
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        source = str(text_path.name)
        
        # Use the appropriate database structure
        if "chunks" in tables:  # Original structure from 1_text_to_vector_db.py
            # Insert chunks directly into the chunks table
            for chunk in chunks:
                # Generate embedding
                embedding = get_embedding(chunk)
                
                # Insert chunk with embedding
                cursor.execute(
                    "INSERT INTO chunks (text, embedding, source) VALUES (?, ?, ?)", 
                    (chunk, embedding.tobytes(), source)
                )
        else:  # Use new structure or create it
            # Create tables if they don't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
            """)
            
            # Insert chunks and their embeddings
            for chunk in chunks:
                # Insert document
                cursor.execute("INSERT INTO documents (source, content) VALUES (?, ?)", (source, chunk))
                document_id = cursor.lastrowid
                
                # Generate embedding
                embedding = get_embedding(chunk)
                
                # Insert embedding
                cursor.execute("INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)", 
                            (document_id, embedding.tobytes()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return False

# Function to create sample documents for the database
def create_sample_documents():
    """Create sample documents and populate the database for demonstration purposes"""
    try:
        # Sample content about financial statements in English and Italian
        sample_texts = [
            {
                "title": "harmonized_financial_statements_overview.txt",
                "content": """
                # Harmonized Financial Statements Overview
                
                Harmonized financial statements are standardized financial reports that follow consistent 
                formatting and calculation rules. They're designed to enable comparison between different 
                entities and time periods.
                
                ## Key Components
                
                1. **Balance Sheet**: Shows assets, liabilities, and equity at a specific point in time
                2. **Income Statement**: Shows revenues, expenses, and profit over a period
                3. **Cash Flow Statement**: Shows cash movements during a period
                4. **Statement of Changes in Equity**: Shows changes in ownership interests
                
                ## XBRL Requirements
                
                The eXtensible Business Reporting Language (XBRL) is commonly used for digital reporting of 
                financial data. Key requirements include:
                
                - Valid taxonomy references
                - Proper element tagging
                - Accurate calculation relationships
                - Complete context information
                - Valid unit declarations
                
                ## Registration Process
                
                To register with BDAP (Database of Public Administrations), organizations typically need to:
                
                1. Create an account on the BDAP platform
                2. Submit identification documents
                3. Receive authentication credentials
                4. Complete the profile with organizational details
                5. Submit test filings before going live
                
                ## Common Controls
                
                Controls performed on harmonized financial statements include:
                
                1. **Completeness checks**: Ensuring all required elements are present
                2. **Mathematical validation**: Verifying calculations are correct
                3. **Cross-statement checks**: Confirming consistency across different statements
                4. **Year-over-year validation**: Checking for reasonable changes from previous periods
                5. **Regulatory compliance**: Ensuring adherence to relevant accounting standards
                """
            },
            {
                "title": "bilanci_armonizzati_panoramica.txt",
                "content": """
                # Panoramica dei Bilanci Armonizzati
                
                I bilanci armonizzati sono documenti finanziari standardizzati che seguono regole coerenti 
                di formattazione e calcolo. Sono progettati per consentire il confronto tra diverse 
                entità e periodi temporali.
                
                ## Componenti Principali
                
                1. **Stato Patrimoniale**: Mostra attività, passività e patrimonio netto a una data specifica
                2. **Conto Economico**: Mostra ricavi, spese e profitto in un periodo
                3. **Rendiconto Finanziario**: Mostra i movimenti di cassa durante un periodo
                4. **Prospetto delle Variazioni del Patrimonio Netto**: Mostra i cambiamenti negli interessi proprietari
                
                ## Requisiti XBRL
                
                L'eXtensible Business Reporting Language (XBRL) è comunemente utilizzato per la reportistica 
                digitale dei dati finanziari. I requisiti chiave includono:
                
                - Riferimenti validi alla tassonomia
                - Corretta etichettatura degli elementi
                - Relazioni di calcolo accurate
                - Informazioni di contesto complete
                - Dichiarazioni di unità valide
                
                ## Processo di Registrazione
                
                Per registrarsi al BDAP (Banca Dati delle Amministrazioni Pubbliche), le organizzazioni 
                tipicamente devono:
                
                1. Creare un account sulla piattaforma BDAP
                2. Presentare documenti di identificazione
                3. Ricevere credenziali di autenticazione
                4. Completare il profilo con i dettagli organizzativi
                5. Inviare documenti di prova prima di andare in produzione
                
                ## Controlli Comuni
                
                I controlli eseguiti sui bilanci armonizzati includono:
                
                1. **Controlli di completezza**: Garantire che tutti gli elementi richiesti siano presenti
                2. **Convalida matematica**: Verificare che i calcoli siano corretti
                3. **Controlli incrociati**: Confermare la coerenza tra i diversi prospetti
                4. **Convalida anno su anno**: Controllare le variazioni ragionevoli rispetto ai periodi precedenti
                5. **Conformità normativa**: Garantire l'aderenza ai principi contabili pertinenti
                """
            },
            {
                "title": "xbrl_technical_requirements.txt",
                "content": """
                # XBRL Technical Requirements for Harmonized Financial Statements
                
                ## Taxonomy Architecture
                
                The XBRL taxonomy for harmonized financial statements follows a modular structure with:
                
                - Core schema containing primary elements
                - Presentation linkbases defining the visual organization
                - Calculation linkbases defining mathematical relationships
                - Definition linkbases establishing element relationships
                - Label linkbases providing human-readable descriptions
                
                ## Validation Rules
                
                XBRL instances must adhere to these key validation rules:
                
                1. Element names must match taxonomy definitions
                2. Context periods must align with reporting periods
                3. Decimal attributes must be specified appropriately
                4. Unit declarations must match element requirements
                5. Calculation relationships must balance mathematically
                
                ## Submission Guidelines
                
                When submitting XBRL instances:
                
                1. File size should not exceed 10MB
                2. XML encoding must be UTF-8
                3. File naming should follow the pattern: [EntityID]_[Period]_[DocumentType].xbrl
                4. Extension taxonomies must be included or referenced
                5. All required contexts must be defined
                
                ## Common Technical Issues
                
                Frequent technical issues include:
                
                1. Inconsistent calculation relationships
                2. Missing or incorrect context references
                3. Incompatible taxonomy versions
                4. Improper handling of signs (positive/negative values)
                5. Incorrect decimal precision specifications
                
                ## Testing and Validation
                
                Before submission, validate XBRL instances using:
                
                1. Taxonomy-specific validation tools
                2. XBRL Formula validation rules
                3. Business rules validation engines
                4. Mathematical consistency checks
                5. Cross-period comparison tools
                """
            },
            {
                "title": "requisiti_tecnici_xbrl.txt",
                "content": """
                # Requisiti Tecnici XBRL per Bilanci Armonizzati
                
                ## Architettura della Tassonomia
                
                La tassonomia XBRL per i bilanci armonizzati segue una struttura modulare con:
                
                - Schema core contenente gli elementi primari
                - Presentation linkbase che definiscono l'organizzazione visiva
                - Calculation linkbase che definiscono le relazioni matematiche
                - Definition linkbase che stabiliscono le relazioni tra elementi
                - Label linkbase che forniscono descrizioni comprensibili
                
                ## Regole di Validazione
                
                Le istanze XBRL devono rispettare queste regole di validazione chiave:
                
                1. I nomi degli elementi devono corrispondere alle definizioni della tassonomia
                2. I periodi di contesto devono allinearsi con i periodi di rendicontazione
                3. Gli attributi decimali devono essere specificati in modo appropriato
                4. Le dichiarazioni di unità devono corrispondere ai requisiti dell'elemento
                5. Le relazioni di calcolo devono essere matematicamente bilanciate
                
                ## Linee Guida per l'Invio
                
                Durante l'invio di istanze XBRL:
                
                1. La dimensione del file non deve superare i 10MB
                2. La codifica XML deve essere UTF-8
                3. La denominazione dei file deve seguire il modello: [EntityID]_[Period]_[DocumentType].xbrl
                4. Le tassonomie di estensione devono essere incluse o referenziate
                5. Tutti i contesti richiesti devono essere definiti
                
                ## Problemi Tecnici Comuni
                
                I problemi tecnici frequenti includono:
                
                1. Relazioni di calcolo incoerenti
                2. Riferimenti di contesto mancanti o errati
                3. Versioni di tassonomia incompatibili
                4. Gestione impropria dei segni (valori positivi/negativi)
                5. Specifiche di precisione decimale errate
                
                ## Test e Validazione
                
                Prima dell'invio, validare le istanze XBRL utilizzando:
                
                1. Strumenti di validazione specifici per la tassonomia
                2. Regole di validazione Formula XBRL
                3. Motori di validazione delle regole di business
                4. Controlli di coerenza matematica
                5. Strumenti di confronto tra periodi
                """
            }
        ]
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database structure
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        # Use the appropriate database structure
        if "chunks" in tables:  # Original structure
            # Create table if not exists
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                embedding BLOB,
                source TEXT
            )
            """)
            
            # Process each sample text
            for sample in sample_texts:
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(sample["content"])
                
                # Store each chunk
                for chunk in chunks:
                    # Generate embedding
                    embedding = get_embedding(chunk)
                    
                    # Insert into database
                    cursor.execute(
                        "INSERT INTO chunks (text, embedding, source) VALUES (?, ?, ?)",
                        (chunk, embedding.tobytes(), sample["title"])
                    )
        else:  # New structure
            # Create tables if not exists
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content TEXT
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
            """)
            
            # Process each sample text
            for sample in sample_texts:
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(sample["content"])
                
                # Store each chunk
                for chunk in chunks:
                    # Insert document
                    cursor.execute(
                        "INSERT INTO documents (source, content) VALUES (?, ?)",
                        (sample["title"], chunk)
                    )
                    document_id = cursor.lastrowid
                    
                    # Generate embedding
                    embedding = get_embedding(chunk)
                    
                    # Insert embedding
                    cursor.execute(
                        "INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)",
                        (document_id, embedding.tobytes())
                    )
        
        conn.commit()
        conn.close()
        
        st.sidebar.success("Sample documents created successfully!")
        return True
    except Exception as e:
        st.sidebar.error(f"Error creating sample documents: {str(e)}")
        return False

# Function to preview PDF
def preview_pdf(pdf_name, language):
    """Show a preview of the PDF"""
    ui = UI_TEXT[language]
    try:
        pdf_path = Path(PDF_DIR) / pdf_name
        if not pdf_path.exists():
            st.error(f"PDF {pdf_name} not found")
            return
            
        # Use st.pdf_viewer component for better PDF rendering
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        # Try two methods of displaying PDFs
        try:
            # Method 1: Using Streamlit's native PDF viewer (preferred)
            st.write(f"**{pdf_name}**")
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf"
            )
            st.pdf_viewer(pdf_bytes, height=600)
        except Exception:
            # Method 2: Fallback to iframe method
            st.write("Using alternate PDF viewer")
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error previewing PDF: {str(e)}")

# Function to download chat history
def download_chat_history(messages, language):
    """Generate a CSV file with chat history and provide download link"""
    ui = UI_TEXT[language]
    if not messages:
        st.warning("No messages to download")
        return
        
    # Create CSV content
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(["Timestamp", "Role", "Content"])
    
    for msg in messages:
        timestamp = msg.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        csv_writer.writerow([timestamp, msg["role"], msg["content"]])
    
    # Provide download link
    csv_str = csv_buffer.getvalue()
    b64 = base64.b64encode(csv_str.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="chat_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">{ui["download_chat"]}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to check if database tables exist
def check_and_create_db_tables():
    """Check if database tables exist and create them if needed"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get existing tables
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        # If the database already has 'chunks' table, it's using the original structure
        if "chunks" in tables:
            # Make sure the structure is correct
            try:
                # Check if embedding column exists
                cursor.execute("PRAGMA table_info(chunks)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if "embedding" not in columns:
                    # Add embedding column if it doesn't exist
                    cursor.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
                    conn.commit()
            except Exception as e:
                st.warning(f"Could not verify chunks table structure: {str(e)}")
        else:
            # Create documents table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                content TEXT
            )
            """)
            
            # Create embeddings table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                embedding BLOB,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
            """)
            conn.commit()
            
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error checking/creating database tables: {str(e)}")
        return False

# Function to fetch and visualize BDAP financial data
def fetch_and_visualize_bdap_data(regione, years_range, data_type, top_n=10):
    """Fetch and visualize financial data from BDAP API"""
    try:
        # 1. Find and download CSV resources
        st.info(f"Searching for data: {regione}, {years_range[0]}-{years_range[-1]}, {data_type}")
        catalogue = []
        progress_bar = st.progress(0)
        
        for i, year in enumerate(years_range):
            try:
                resource = find_csv_resource(year, regione, data_type)
                catalogue.append(resource)
                progress_bar.progress((i + 1) / len(years_range))
            except LookupError as e:
                st.warning(f"Warning: {str(e)}")
                continue
        
        if not catalogue:
            st.error(f"No data found for {regione} in the selected years")
            return None
        
        st.success(f"Found {len(catalogue)} CSV files")
        
        # 2. Download and concatenate data
        frames = []
        data_progress = st.progress(0)
        
        for i, meta in enumerate(catalogue):
            try:
                st.write(f"Downloading {meta['year']} data...")
                
                # Use multiple retry attempts with shorter timeouts
                for attempt in range(3):
                    try:
                        raw = requests.get(meta["url"], timeout=30).content
                        break
                    except (requests.ConnectionError, requests.Timeout):
                        if attempt < 2:
                            st.warning(f"Download attempt {attempt+1}/3 failed. Retrying...")
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise
                
                df = pd.read_csv(io.BytesIO(raw), low_memory=False)
                df["year"] = meta["year"]
                frames.append(df)
                data_progress.progress((i + 1) / len(catalogue))
                
            except Exception as e:
                st.error(f"Error downloading {meta['year']} data: {str(e)}")
                st.info("Skipping this year and continuing with others...")
                continue
        
        # 3. Combine and clean data
        if not frames:
            st.error("Could not download any data. Please try again later.")
            return None
            
        data = pd.concat(frames, ignore_index=True).rename(columns=str.lower)
        
        # Handle different column naming conventions
        if "importo" in data.columns:
            amount_col = "importo"
        elif "valore" in data.columns:
            amount_col = "valore"
        else:
            # Try to find a column that might contain monetary values
            numeric_cols = data.select_dtypes(include=['number']).columns
            if numeric_cols.empty:
                st.error("No numeric column found to use as amount")
                return None
            amount_col = numeric_cols[0]
            st.warning(f"Using column '{amount_col}' as amount column")
        
        # Extract title columns
        if "cod_titolo" in data.columns and "descr_titolo" in data.columns:
            title_code_col = "cod_titolo"
            title_desc_col = "descr_titolo"
        else:
            # Look for columns that might contain title information
            code_candidates = [col for col in data.columns if "cod" in col.lower() and "titolo" in col.lower()]
            desc_candidates = [col for col in data.columns if "desc" in col.lower() and "titolo" in col.lower()]
            
            if code_candidates and desc_candidates:
                title_code_col = code_candidates[0]
                title_desc_col = desc_candidates[0]
            else:
                st.error("Could not identify title code and description columns")
                return None
        
        # Clean data
        keep = ["year", title_code_col, title_desc_col, amount_col]
        data = data[keep].dropna(subset=[amount_col])
        
        # Return cleaned data and column names for visualization
        return {
            "data": data,
            "amount_col": amount_col,
            "title_code_col": title_code_col,
            "title_desc_col": title_desc_col
        }
    
    except Exception as e:
        st.error(f"Error fetching BDAP data: {str(e)}")
        return None

# Function to visualize BDAP financial data
def visualize_bdap_data(data_dict, regione, data_type, top_n=10):
    """Create visualizations from BDAP financial data"""
    if not data_dict:
        return
    
    data = data_dict["data"]
    amount_col = data_dict["amount_col"]
    title_code_col = data_dict["title_code_col"]
    title_desc_col = data_dict["title_desc_col"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Time trend visualization
        st.subheader("Time Trend Analysis")
        trend = (data.groupby("year", as_index=False)
                    .agg({amount_col: "sum"})
                    .rename(columns={amount_col: "total_amount"}))
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(trend["year"], trend["total_amount"], marker="o", linewidth=2)
        ax1.set_title(f"{regione} - {data_type.capitalize()} ({trend['year'].min()}-{trend['year'].max()})")
        ax1.set_ylabel("€")
        ax1.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        # 2. Structure visualization (latest year)
        st.subheader("Structure Analysis (Latest Year)")
        latest_year = data["year"].max()
        latest = data[data["year"] == latest_year]
        
        struct = (latest.groupby([title_code_col, title_desc_col], as_index=False)
                        .agg({amount_col: "sum"})
                        .sort_values(amount_col, ascending=False)
                        .head(top_n))
        
        # Calculate percentages
        total = struct[amount_col].sum()
        struct["percentage"] = struct[amount_col] / total * 100
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bars = ax2.barh(struct[title_desc_col], struct[amount_col])
        ax2.set_title(f"{regione} - Top {top_n} {data_type} Categories ({latest_year})")
        ax2.set_xlabel("€")
        ax2.invert_yaxis()  # Display from top to bottom in descending order
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax2.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f"{struct['percentage'].iloc[i]:.1f}%", 
                    va='center')
            
        plt.tight_layout()
        st.pyplot(fig2)
    
    # 3. Data table view
    st.subheader("Data Summary")
    st.dataframe(struct[[title_desc_col, amount_col, "percentage"]], hide_index=True)
    
    # 4. Download button for the data
    csv = struct.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"{regione}_{data_type}_{latest_year}.csv",
        mime="text/csv"
    )

# Function to show document statistics
def visualize_embeddings(language):
    """Visualize document embeddings with dimensionality reduction"""
    ui = UI_TEXT[language]
    
    try:
        # Check if database exists
        if not Path(DB_PATH).exists():
            st.warning(ui["db_not_found"])
            return
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all embeddings and their metadata
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        embeddings = []
        doc_sources = []
        doc_ids = []
        doc_contents = []
        
        # Handle different database structures
        if "documents" in tables and "embeddings" in tables:
            # Modern structure with separate tables
            query = """
                SELECT d.id, d.content, d.source, e.embedding 
                FROM documents d 
                JOIN embeddings e ON d.id = e.document_id
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                doc_id, content, source, embedding_bytes = row
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings.append(embedding)
                doc_sources.append(source)
                doc_ids.append(doc_id)
                doc_contents.append(content[:100] + "..." if len(content) > 100 else content)
                
        elif "paragraphs" in tables:
            # Our current structure from 1_text_to_vector_db.py
            # Check if the paragraphs table has the expected columns
            columns = [col[1] for col in cursor.execute("PRAGMA table_info(paragraphs)").fetchall()]
            if "content" in columns and "source" in columns and "embedding" in columns:
                print(f"Found paragraphs table with correct columns: {columns}")
                query = "SELECT id, content, source, embedding FROM paragraphs"
                cursor.execute(query)
                rows = cursor.fetchall()
                
                for row in rows:
                    doc_id, content, source, embedding_bytes = row
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    embeddings.append(embedding)
                    doc_sources.append(source)
                    doc_ids.append(doc_id)
                    doc_contents.append(content[:100] + "..." if len(content) > 100 else content)
                    
        elif "chunks" in tables:
            # Legacy structure with chunks table
            if "embedding" in [col[1] for col in cursor.execute("PRAGMA table_info(chunks)").fetchall()]:
                query = "SELECT id, content, source, embedding FROM chunks"
                cursor.execute(query)
                rows = cursor.fetchall()
                
                for row in rows:
                    doc_id, content, source, embedding_bytes = row
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    embeddings.append(embedding)
                    doc_sources.append(source)
                    doc_ids.append(doc_id)
                    doc_contents.append(content[:100] + "..." if len(content) > 100 else content)
        
        conn.close()
        
        if not embeddings:
            st.warning("No embeddings found in the database.")
            st.info("Please ensure your database is properly initialized. If you've added documents, try running the 1_text_to_vector_db.py script to create embeddings.")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create color mapping based on document source with distinct colors
        unique_sources = list(set(doc_sources))
        source_to_color = {}
        
        # Define a colormap with more distinct colors
        colormap = [
            'rgba(255, 50, 50, 0.8)',   # Red
            'rgba(50, 50, 255, 0.8)',    # Blue
            'rgba(50, 205, 50, 0.8)',    # Green
            'rgba(255, 165, 0, 0.8)',    # Orange
            'rgba(128, 0, 128, 0.8)',    # Purple
            'rgba(0, 139, 139, 0.8)',    # Teal
            'rgba(255, 20, 147, 0.8)',   # Pink
            'rgba(184, 134, 11, 0.8)',   # Golden Brown
            'rgba(0, 191, 255, 0.8)',    # Deep Sky Blue
            'rgba(139, 69, 19, 0.8)'     # Saddle Brown
        ]
        
        # Map each unique source to a distinct color
        for i, source in enumerate(unique_sources):
            color_idx = i % len(colormap)  # Cycle through colors if more sources than colors
            source_to_color[source] = colormap[color_idx]
                
        # Color list for each embedding point
        colors = [source_to_color[source] for source in doc_sources]
        
        # Dimensionality reduction options
        available_methods = ["PCA", "t-SNE"]
        if UMAP_AVAILABLE:
            available_methods.append("UMAP")
            
        dim_reduction_method = st.selectbox(
            ui["vis_dim_reduction"],
            options=available_methods,
            index=0
        )
        
        # Apply dimensionality reduction
        with st.spinner("Reducing dimensionality..."):
            if dim_reduction_method == "PCA":
                # PCA - fast but less effective for complex relationships
                pca = PCA(n_components=2)
                reduced_embeddings = pca.fit_transform(embeddings_array)
            elif dim_reduction_method == "t-SNE":
                # t-SNE - better for clusters but slower
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
                reduced_embeddings = tsne.fit_transform(embeddings_array)
            else:  # UMAP - only if available
                if UMAP_AVAILABLE:
                    # UMAP - good balance between performance and quality
                    umap_reducer = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings_array)-1))
                    reduced_embeddings = umap_reducer.fit_transform(embeddings_array)
                else:
                    # Fallback to PCA if UMAP not available
                    st.warning("UMAP is not installed. Using PCA instead.")
                    pca = PCA(n_components=2)
                    reduced_embeddings = pca.fit_transform(embeddings_array)
        
        # Create tooltip texts
        hover_texts = [f"ID: {doc_id}<br>Source: {source}<br>Content: {content}" 
                      for doc_id, source, content in zip(doc_ids, doc_sources, doc_contents)]
        
        # Create a legend with document types
        pdf_chunks = ui["pdf_chunks"]
        text_chunks = ui["text_chunks"]
        
        # Create plot
        fig = go.Figure()
        
        # We've replaced this block with the new approach that creates a trace per source
        
        # Group points by source and add a trace for each source with its own color
        for source in unique_sources:
            source_indices = [i for i, s in enumerate(doc_sources) if s == source]
            if source_indices:
                # Extract just the filename from path for the legend
                source_name = source.split('/')[-1] if '/' in source else source
                # Truncate long names
                if len(source_name) > 25:
                    source_name = source_name[:22] + '...'
                    
                fig.add_trace(go.Scatter(
                    x=reduced_embeddings[source_indices, 0],
                    y=reduced_embeddings[source_indices, 1],
                    mode='markers',
                    marker=dict(color=source_to_color[source], size=10),
                    text=[hover_texts[i] for i in source_indices],
                    hoverinfo='text',
                    name=source_name
                ))
        
        # Update layout
        fig.update_layout(
            title=ui["vis_title"],
            xaxis=dict(title=f"{dim_reduction_method} Component 1"),
            yaxis=dict(title=f"{dim_reduction_method} Component 2"),
            hovermode='closest'
        )
        
        # Display the visualization
        st.plotly_chart(fig, use_container_width=True)
        
        # Add descriptions
        st.markdown(ui["vis_description"])
        
        # Display color-coded document sources for reference
        if unique_sources:
            st.subheader("Document Sources")
            for source in unique_sources:
                # Extract just the filename from path
                source_name = source.split('/')[-1] if '/' in source else source
                # Display with color-matched markdown
                color = source_to_color[source].replace('rgba', 'rgb').replace(', 0.8)', ')')
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>■</span> {source_name}", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error visualizing embeddings: {str(e)}")

def show_document_statistics(language):
    """Show statistics about the documents in the database"""
    ui = UI_TEXT[language]
    
    try:
        # Check if database exists
        if not Path(DB_PATH).exists():
            st.warning(ui["db_not_found"])
            st.info(f"To create the database:\n1. Upload PDFs in the '{ui['tab_docs']}' tab\n2. Or run the scripts 0_pdf_to_text.py and 1_text_to_vector_db.py")
            return
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check database structure
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [t[0] for t in tables]
        
        # Add a document selector at the top of the visualization tab
        st.subheader("Document Selection")
        st.info("Select documents to visualize. You can select multiple documents and see them color-coded in the interactive plots.")
        
        # Initialize variables
        doc_count = 0
        source_counts = []
        results = []
        all_sources = []
        
        # Get all available sources based on the database structure
        if "paragraphs" in tables:
            cursor.execute("SELECT DISTINCT source FROM paragraphs")
            all_sources = [row[0] for row in cursor.fetchall()]
        elif "chunks" in tables:
            cursor.execute("SELECT DISTINCT source FROM chunks")
            all_sources = [row[0] for row in cursor.fetchall()]
        elif "documents" in tables:
            cursor.execute("SELECT DISTINCT source FROM documents")
            all_sources = [row[0] for row in cursor.fetchall()]
        
        # Multiple document selection
        source_options = all_sources
        st.write("Select documents to visualize:")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_sources = st.multiselect(
                "Documents", 
                source_options,
                default=source_options[:min(3, len(source_options))] if source_options else None,
                label_visibility="collapsed"
            )
        with col2:
            select_all = st.checkbox("Select All", value=False)
            
        # Handle select all option
        if select_all:
            selected_sources = source_options
        
        # Show document selection information
        if selected_sources:
            st.caption(f"Showing {len(selected_sources)} documents with interactive visualizations. Hover over points to see document details, click and drag to zoom, and double-click to reset the view.")
            
            # Initialize doc_colors for backward compatibility with any code that might still use it
            doc_colors = {}
        
        # Add a filter clause for the SQL queries based on selected documents
        source_filter = ""
        if selected_sources:
            source_list = "', '".join(selected_sources)
            source_filter = f"WHERE source IN ('{source_list}')"
        
        # Handle different database structures
        if "paragraphs" in tables:  # Current structure from text_to_vector_db.py
            # Get document count - apply filter if a specific document is selected
            count_query = f"SELECT COUNT(*) FROM paragraphs {source_filter}"
            cursor.execute(count_query)
            doc_count = cursor.fetchone()[0]
            
            if doc_count == 0:
                st.warning("No paragraphs found in the database.")
                st.info(f"Please upload PDFs in the '{ui['tab_docs']}' tab or run the preprocessing scripts to add documents to the database.")
                conn.close()
                return
            
            # Get source counts based on selected sources
            if selected_sources:
                source_list = "', '".join(selected_sources)
                cursor.execute(f"SELECT source, COUNT(*) FROM paragraphs WHERE source IN ('{source_list}') GROUP BY source")
            else:
                cursor.execute("SELECT source, COUNT(*) FROM paragraphs GROUP BY source")
            source_counts = cursor.fetchall()
            
            # Get embeddings for visualization (limit to 200 to avoid performance issues)
            viz_query = f"SELECT source, embedding FROM paragraphs {source_filter} LIMIT 200"
            cursor.execute(viz_query)
            results = cursor.fetchall()
            
        elif "chunks" in tables:  # Original structure from 1_text_to_vector_db.py
            # Get document count with filter
            count_query = f"SELECT COUNT(*) FROM chunks {source_filter}"
            cursor.execute(count_query)
            doc_count = cursor.fetchone()[0]
            
            if doc_count == 0:
                st.warning("No chunks found in the database.")
                st.info(f"Please upload PDFs in the '{ui['tab_docs']}' tab or run the preprocessing scripts to add documents to the database.")
                conn.close()
                return
            
            # Get source counts based on selected sources
            if selected_sources:
                source_list = "', '".join(selected_sources)
                cursor.execute(f"SELECT source, COUNT(*) FROM chunks WHERE source IN ('{source_list}') GROUP BY source")
            else:
                cursor.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source")
            source_counts = cursor.fetchall()
            
            # Get embeddings for visualization with filter
            viz_query = f"SELECT source, embedding FROM chunks {source_filter} LIMIT 200"
            cursor.execute(viz_query)
            results = cursor.fetchall()
            
        elif "documents" in tables:  # New structure
            # Get document count with filter
            count_query = f"SELECT COUNT(*) FROM documents {source_filter}"
            cursor.execute(count_query)
            doc_count = cursor.fetchone()[0]
            
            if doc_count == 0:
                st.warning("No documents found in the database.")
                st.info(f"Please upload PDFs in the '{ui['tab_docs']}' tab or run the preprocessing scripts to add documents to the database.")
                conn.close()
                return
            
            # Get source counts based on selected sources
            if selected_sources:
                source_list = "', '".join(selected_sources)
                cursor.execute(f"SELECT source, COUNT(*) FROM documents WHERE source IN ('{source_list}') GROUP BY source")
            else:
                cursor.execute("SELECT source, COUNT(*) FROM documents GROUP BY source")
            source_counts = cursor.fetchall()
            
            if "embeddings" in tables:
                # Get embeddings for visualization with filter
                if selected_sources:
                    source_list = "', '".join(selected_sources)
                    viz_query = f"""SELECT d.source, e.embedding 
                        FROM embeddings e 
                        JOIN documents d ON e.document_id = d.id 
                        WHERE d.source IN ('{source_list}')
                        LIMIT 200"""
                else:
                    viz_query = """SELECT d.source, e.embedding 
                        FROM embeddings e 
                        JOIN documents d ON e.document_id = d.id 
                        LIMIT 200"""
                cursor.execute(viz_query)
                results = cursor.fetchall()
        else:
            st.warning(f"Unknown database structure. Available tables: {', '.join(tables)}")
            st.info(f"Please run the preprocessing scripts to create the proper database structure.")
            conn.close()
            return
        
        # Create dataframe for visualization
        df = pd.DataFrame(source_counts, columns=["Source", "Chunks"])
        if not df.empty:
            df["Source"] = df["Source"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)  # Truncate long names
        
        # Display statistics with filtering information
        st.subheader(ui["doc_stats"])
        col1, col2 = st.columns(2)
        
        # Show filter information if specific documents are selected
        if selected_sources:
            if len(selected_sources) == 1:
                col1.metric("Document", selected_sources[0].split('/')[-1])
            else:
                col1.metric("Documents Selected", len(selected_sources))
            col1.metric("Paragraphs/Chunks", doc_count)
            # Display the number of unique documents from source_counts
            col1.metric("Unique Sources", len(source_counts))
        else:
            col1.metric("Total Documents", len(all_sources))
            col1.metric("Total Paragraphs/Chunks", doc_count)
            # Display the number of unique documents from source_counts
            col1.metric("Unique Sources", len(source_counts))
        
        # Create bar chart
        if not df.empty:
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Chunks", y="Source", data=df, ax=ax)
                ax.set_title("Chunks per Document")
                st.pyplot(fig)
        
        # Get embeddings for visualization
        if selected_sources:
            if len(selected_sources) == 1:
                st.subheader(f"Vector Space Visualization for '{selected_sources[0].split('/')[-1]}'")
            else:
                st.subheader(f"Vector Space Visualization ({len(selected_sources)} Documents)")
        else:
            st.subheader("Vector Space Visualization (All Documents)")
        
        if results:
            # Process embeddings with more robust handling
            sources = []
            embeddings = []
            
            for source, emb_bytes in results:
                sources.append(source)
                # Convert bytes to numpy array
                emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                embeddings.append(emb_array)
                
            # Add a separator to visually distinguish the visualization sections
            st.markdown("---")
            
            if not embeddings:
                # Display message if visualizations are empty
                if selected_sources:
                    st.warning("No data available for the selected documents. Please choose different documents or upload new ones.")
                else:
                    st.info("No documents selected for visualization. Please select at least one document above.")
            elif embeddings:
                # Create a DataFrame with embeddings
                embeddings_array = np.vstack(embeddings)
                
                # 1. PCA Visualization
                st.write("##### PCA Visualization of Embeddings")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Perform PCA
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(embeddings_array)
                    
                    # Create a more robust dataframe for plotting with explicit document categories
                    doc_data = []
                    
                    # Process embeddings and create a list of dictionaries for the dataframe
                    for i, (source, vector) in enumerate(zip(sources, pca_result)):
                        doc_name = source.split('/')[-1].split('.')[0]
                        doc_data.append({
                            'x': vector[0],
                            'y': vector[1],
                            'source': doc_name,  # Short name for display
                            'full_source': source,  # Full path
                            'doc_id': i % len(set(sources))  # Ensure unique IDs for coloring
                        })
                    
                    # Create DataFrame from the processed data
                    pca_df = pd.DataFrame(doc_data)
                    
                    # Create an interactive Plotly scatter plot with explicit color mapping
                    # Use Plotly's Vivid color sequence for maximum contrast between documents
                    fig_pca = px.scatter(
                        pca_df, 
                        x='x', 
                        y='y', 
                        color='source',  # Use source for coloring
                        hover_name='source',
                        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2', 'source': 'Document'},
                        title='PCA of Document Embeddings',
                        color_discrete_sequence=px.colors.qualitative.Vivid,  # Use Vivid for better contrast
                        height=500
                    )
                    
                    # Customize the layout
                    fig_pca.update_layout(
                        legend_title_text='Document Source',
                        legend=dict(
                            itemsizing='constant',  # Make legend items the same size
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            font=dict(size=10),  # Smaller font for more documents
                            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background for better readability
                        ),
                        plot_bgcolor='white',
                        hovermode='closest'
                    )
                    
                    # Add grid lines
                    fig_pca.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig_pca.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    
                    # Update markers
                    fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
                    
                    # Use the custom legend we created above for consistent coloring
                    display_names = {}
                    for source in pca_df['full_source'].unique():
                        display_names[source] = source.split('/')[-1].split('.')[0]
                    
                    # Update hover template to show full document name
                    hover_template = '<b>%{hovertext}</b><br>'
                    fig_pca.update_traces(hovertemplate=hover_template)
                    
                    # Display the interactive plot
                    st.plotly_chart(fig_pca, use_container_width=True)
                

                
                # 2. t-SNE Visualization
                with col2:
                    st.write("##### t-SNE Visualization of Embeddings")
                    # Perform t-SNE
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_array)-1))
                    tsne_result = tsne.fit_transform(embeddings_array)
                    
                    # Create t-SNE dataframe with the same robust structure as PCA
                    tsne_data = []
                    
                    # Process embeddings and create a list of dictionaries for the dataframe
                    for i, (source, vector) in enumerate(zip(sources, tsne_result)):
                        doc_name = source.split('/')[-1].split('.')[0]
                        tsne_data.append({
                            'x': vector[0],
                            'y': vector[1],
                            'source': doc_name,  # Short name for display
                            'full_source': source,  # Full path
                            'doc_id': i % len(set(sources))  # Ensure unique IDs for coloring
                        })
                    
                    # Create DataFrame from the processed data
                    tsne_df = pd.DataFrame(tsne_data)
                    
                    # Create an interactive Plotly scatter plot for t-SNE with same color scheme as PCA
                    fig_tsne = px.scatter(
                        tsne_df, 
                        x='x', 
                        y='y', 
                        color='source',  # Use source for coloring
                        hover_name='source',
                        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'source': 'Document'},
                        title='t-SNE of Document Embeddings',
                        color_discrete_sequence=px.colors.qualitative.Vivid,  # Use same color scheme as PCA
                        height=500
                    )
                    
                    # Customize the layout
                    fig_tsne.update_layout(
                        legend_title_text='Document Source',
                        legend=dict(
                            itemsizing='constant',  # Make legend items the same size
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            font=dict(size=10),  # Smaller font for more documents
                            bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent background
                        ),
                        plot_bgcolor='white',
                        hovermode='closest'
                    )
                    
                    # Add grid lines
                    fig_tsne.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig_tsne.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    
                    # Update markers
                    fig_tsne.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
                    
                    # Update hover template to show full document name
                    fig_tsne.update_traces(hovertemplate='<b>%{hovertext}</b><br>')
                    
                    # Display the interactive plot
                    st.plotly_chart(fig_tsne, use_container_width=True)
                    
                # Show explained variance of PCA
                st.write(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")
        
        conn.close()
    except Exception as e:
        st.error(f"Error analyzing documents: {str(e)}")

# Function to display RAG settings form
def display_rag_settings(language):
    """Display and manage RAG settings"""
    ui = UI_TEXT[language]
    st.subheader(ui["rag_settings"])
    
    # Initialize default settings in session state if not present
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
    if "similarity_top_k" not in st.session_state:
        st.session_state.similarity_top_k = 5
    
    # Create a form for settings
    with st.form(key="rag_settings_form"):
        chunk_size = st.slider(
            ui["chunk_size"],
            min_value=100, max_value=2000, value=st.session_state.chunk_size, step=100
        )
        
        chunk_overlap = st.slider(
            ui["chunk_overlap"],
            min_value=0, max_value=500, value=st.session_state.chunk_overlap, step=50
        )
        
        similarity_top_k = st.slider(
            ui["similarity_top_k"],
            min_value=1, max_value=10, value=st.session_state.similarity_top_k, step=1
        )
        
        submit_button = st.form_submit_button("Apply Settings")
        
        if submit_button:
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.similarity_top_k = similarity_top_k
            st.success("Settings applied successfully")
    
    return {
        "chunk_size": st.session_state.chunk_size,
        "chunk_overlap": st.session_state.chunk_overlap,
        "similarity_top_k": st.session_state.similarity_top_k
    }

# Function to display response feedback options
def display_feedback(language):
    """Display feedback options for the response"""
    ui = UI_TEXT[language]
    
    st.markdown(f"### {ui['feedback_title']}")
    col1, col2 = st.columns(2)
    
    if col1.button(ui["feedback_yes"], key="feedback_yes"):
        st.session_state.feedback = "positive"
        st.success(ui["feedback_thanks"])
        
    if col2.button(ui["feedback_no"], key="feedback_no"):
        st.session_state.feedback = "negative"
        st.success(ui["feedback_thanks"])

# Function to display post-response options
def display_post_response_options(language):
    """Display options after a response is given"""
    ui = UI_TEXT[language]
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button(ui["continue_questions"], key="continue_questions"):
        st.session_state.post_response_action = "continue_questions"
        return "continue_questions"
        
    if col2.button(ui["ask_more"], key="ask_more"):
        st.session_state.post_response_action = "ask_more"
        return "ask_more"
        
    if col3.button(ui["restart_chat"], key="restart_chat"):
        st.session_state.post_response_action = "restart_chat"
        # Clear chat history
        st.session_state.messages = []
        st.rerun()
        
    return None

# Function to display sample questions
def display_sample_questions(language="Italiano"):
    """Display sample questions for the user to try"""
    ui = UI_TEXT[language]
    st.subheader(ui["sample_title"])
    st.caption(ui["sample_click"])
    
    # Container for sample questions
    container = st.container()
    
    # Display sample questions in a grid layout
    cols = container.columns(2)  # Adjust number of columns as needed
    
    # Get sample questions for the selected language
    questions = SAMPLE_QUESTIONS[language]
    
    # Distribute questions across columns
    for i, question in enumerate(questions):
        col_idx = i % 2  # Determine which column to place the question
        with cols[col_idx]:
            if st.button(question, key=f"sample_{i}"):
                return question
    
    return None

# Main function
def main():
    """Main function for the Streamlit app"""
    # Make sure we have access to the os module
    import os
    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize other session state variables
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "post_response_action" not in st.session_state:
        st.session_state.post_response_action = None
    if "show_pdf_preview" not in st.session_state:
        st.session_state.show_pdf_preview = False
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = None
    if "show_rag_settings" not in st.session_state:
        st.session_state.show_rag_settings = False
    if "show_doc_insights" not in st.session_state:
        st.session_state.show_doc_insights = False
        
    # Select language
    language = st.sidebar.selectbox(
        "Language / Lingua",
        options=["English", "Italiano"],
        index=1  # Default to Italian
    )
    ui = UI_TEXT[language]
    
    # Database diagnostics and repair
    st.sidebar.subheader("Database Status")
    if Path(DB_PATH).exists():
        try:
            # Basic repair and validation
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check tables and structure
            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            tables = [t[0] for t in tables]
            
            # Check database structure
            if "paragraphs" in tables:  # Structure from updated 1_text_to_vector_db.py
                # Paragraphs structure (actual structure in the vector db script)
                st.session_state.db_table_structure = "paragraphs"
                doc_count = cursor.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
                
                if doc_count > 0:
                    st.sidebar.success(f"Using bilanci_vectors.db - Found {doc_count} paragraphs")
                else:
                    st.sidebar.warning("Database exists but contains no data!")
                    st.sidebar.info("""
                    To add content to the database:
                    1. Make sure you have PDF files in the 'rules' folder
                    2. Run 0_pdf_to_text.py to extract text from PDFs
                    3. Run 1_text_to_vector_db.py again to process the text files
                    """)
            elif "chunks" in tables:  # Original structure from 1_text_to_vector_db.py
                # Original table structure from the 1_text_to_vector_db.py script
                st.session_state.db_table_structure = "original"
                doc_count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                
                if doc_count > 0:
                    st.sidebar.success(f"Using bilanci_vectors.db - Found {doc_count} chunks")
                else:
                    st.sidebar.warning("Database exists but contains no data!")
                    st.sidebar.info("""
                    To add content to the database:
                    1. Make sure you have PDF files in the 'rules' folder
                    2. Run 0_pdf_to_text.py to extract text from PDFs
                    3. Run 1_text_to_vector_db.py again to process the text files
                    """)
                        
            elif "documents" in tables:
                # New table structure we're expecting
                st.session_state.db_table_structure = "new"
                doc_count = cursor.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                
                if doc_count > 0:
                    st.sidebar.success(f"Using bilanci_vectors.db - Found {doc_count} documents")
                else:
                    st.sidebar.warning("Database exists but contains no data!")
                    st.sidebar.info("""
                    To add content to the database:
                    1. Make sure you have PDF files in the 'rules' folder
                    2. Run 0_pdf_to_text.py to extract text from PDFs
                    3. Run 1_text_to_vector_db.py again to process the text files
                    """)
            else:
                st.session_state.db_table_structure = "unknown"
                st.sidebar.warning("Unknown structure in bilanci_vectors.db. Tables: " + ", ".join(tables))
                
                st.sidebar.info("Please run 0_pdf_to_text.py and 1_text_to_vector_db.py to properly initialize the database.")
                    
            conn.close()
        except Exception as e:
            st.sidebar.error(f"Database validation error: {str(e)}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header(ui["header"])
        
        # Select API source
        api_source = st.radio(
            ui["api_source"],
            options=["groq", "ollama"],
            format_func=lambda x: ui[x],
            horizontal=True,
            key="api_source"
        )
        
        # Select LLM model based on API source
        if api_source == "groq":
            model_options = GROQ_MODELS
            if not os.getenv("GROQ_API_KEY"):
                # Try to read from file if environment variable not set
                try:
                    with open(".groq_api_key", "r") as f:
                        api_key = f.read().strip()
                    os.environ["GROQ_API_KEY"] = api_key
                except:
                    st.warning("No Groq API key found. Please enter your API key or switch to Ollama.")
                    api_key = st.text_input(ui["api_key"], type="password")
                    if api_key:
                        os.environ["GROQ_API_KEY"] = api_key
                        # Save to file for future use
                        with open(".groq_api_key", "w") as f:
                            f.write(api_key)
        else:  # ollama
            # Get dynamically available Ollama models
            model_options = get_available_ollama_models()
            st.info("Make sure Ollama is running with 'ollama serve' command. Install models with 'ollama pull <model-name>'")
            
        # Select model from appropriate options
        model_name = model_options[st.selectbox(
            ui["select_model"],
            options=list(model_options.keys()),
            key="model_selector"
        )]
        
        # API key input
        groq_api_key = st.text_input(ui["api_key"], type="password")
        
        # Database status
        if Path(DB_PATH).exists():
            st.success(ui["db_found"])
        else:
            st.error(ui["db_not_found"])
            st.info(ui["db_instructions"])
        
        # Core functionality options
        st.subheader("Options")
        
        # Enable web search toggle
        enable_web_search = st.checkbox(ui["enable_web"], value=False)
        
        # Enable text-to-speech toggle
        enable_tts = st.checkbox(ui["enable_tts"], value=False)
        
        # Show thinking process toggle
        show_thinking = st.checkbox(ui["show_thinking"], value=False)
        
        # Download chat history
        if st.session_state.messages:
            st.subheader("Actions")
            download_chat_history(st.session_state.messages, language)
    
    # Title and header
    st.title(ui["main_title"])
    st.markdown(ui["description"])
    
    # Create tabs for better organization
    tab_chat, tab_docs, tab_vis, tab_data, tab_formulator, tab_settings = st.tabs([
        ui["tab_chat"], 
        ui["tab_docs"], 
        ui["tab_vis"],
        ui["tab_data"],
        "Data Formulator",
        ui["tab_settings"]
    ])
    
    # Initialize user_input outside of tabs
    user_input = None
    selected_question = None

    # Tab 1: Chat Interface
    with tab_chat:
        # Display chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Play audio if available
                if "audio" in message and message["audio"] and enable_tts:
                    audio_bytes = base64.b64decode(message["audio"])
                    st.audio(audio_bytes, format="audio/mp3")
        
        # Post-response options if there are messages
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            # Show feedback options
            if st.session_state.feedback is None:
                display_feedback(language)
            
            # Show post-response actions
            action = display_post_response_options(language)
            if action == "continue_questions":
                st.rerun()  # Refresh to show sample questions again
        
        # Display sample questions
        selected_question = display_sample_questions(language)
    
    # Tab 2: Document Viewer
    with tab_docs:
        # Available documents
        pdfs = list_available_pdfs()
        if pdfs:
            st.subheader(ui["docs_available"])
            selected_pdf_to_view = st.selectbox("Select a document to view", pdfs)
            if selected_pdf_to_view:
                preview_pdf(selected_pdf_to_view, language)
        else:
            st.warning(ui["no_docs"])
        
        # Upload PDF directly in the documents tab
        st.subheader(ui["upload_pdf"])
        st.caption(ui["upload_instructions"])
        uploaded_file = st.file_uploader("PDF", type="pdf", key="doc_tab_uploader", label_visibility="collapsed")
        if uploaded_file is not None:
            upload_pdf(uploaded_file, language)
            st.rerun()  # Refresh the page to show the newly uploaded PDF
    
    # Tab 3: Vector Visualizations
    with tab_vis:
        # Display vector embeddings visualization
        visualize_embeddings(language)
            
    # Tab 4: Financial Data Analysis
    with tab_data:
        st.header(ui["data_header"])
        st.markdown(ui["data_description"])
        
        # API Selection
        api_source = st.radio(
            ui["api_source"],
            ["primary", "alternative"],
            format_func=lambda x: ui["primary_api"] if x == "primary" else ui["alternative_api"],
            horizontal=True
        )
        
        if api_source == "primary":
            # Original BDAP CKAN API for local budgets
            # Data input form
            with st.form(key="bdap_data_form"):
                # 1. Region selection
                regions = [
                    "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna", 
                    "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche", 
                    "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia", "Toscana", 
                    "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"
                ]
                regione = st.selectbox(ui["region_select"], regions, index=4)  # Default to Emilia-Romagna
                
                # 2. Data type selection
                data_types = ["ENTRATE", "SPESE"]
                data_type_labels = ["Revenue/Entrate", "Expenditure/Spese"]
                data_type = st.selectbox(ui["data_type"], data_types, format_func=lambda x: data_type_labels[data_types.index(x)])
                
                # 3. Year range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_year = st.selectbox("Start Year", range(2009, 2024), index=0)
                with col2:
                    end_year = st.selectbox("End Year", range(2009, 2024), index=14)  # Default to 2023
                
                if start_year > end_year:
                    st.error("Start year cannot be greater than end year")
                    years_range = range(end_year, start_year + 1)
                else:
                    years_range = range(start_year, end_year + 1)
                
                # 4. Top N categories
                top_n = st.slider(ui["top_n"], min_value=3, max_value=20, value=10, step=1)
                
                # Submit button
                submitted = st.form_submit_button(ui["fetch_data"])
                
        else: # Alternative API
            # Show an explanation of this API option with clearer guidance
            st.info("The alternative BDAP API provides access to a wider range of Italian government financial datasets. Click the button below to load the available datasets (this may take a few seconds).") 
            
            # Add a warning about potential slowdowns
            st.caption("Note: The alternative API might be slower or timeout during peak hours. If you encounter issues, please try the primary API.")
        
            # Load datasets button to make fetching on-demand instead of automatic
            datasets_loaded = False
            if "all_datasets" in st.session_state:
                datasets_loaded = True
                all_datasets = st.session_state.all_datasets
                
                # Show dataset count
                st.success(f"Loaded {len(all_datasets)} available datasets from BDAP alternative API.")
            
            if not datasets_loaded and st.button("Load Available Datasets"):
                # Get datasets from alternative API
                try:
                    # Add a loading indicator for dataset list
                    with st.spinner("Loading available datasets from BDAP alternative API..."):
                        # Cache the dataset list to improve performance
                        @st.cache_data(ttl=3600) # Cache for 1 hour
                        def get_cached_datasets(filter_keyword=None):
                            return get_alt_datasets(filter_keyword)
                        
                        # Try to get the list of datasets
                        all_datasets = get_cached_datasets()
                        
                        # Store in session state
                        st.session_state.all_datasets = all_datasets
                    
                    # Show success message and rerun to refresh UI
                    st.success(f"Successfully loaded {len(all_datasets)} datasets")
                    st.rerun()
                except (requests.ConnectionError, requests.Timeout) as e:
                    st.error(f"Connection error when loading datasets: The BDAP API server may be unavailable.")
                    st.info("Please try again later or use the primary API option instead.")
                except Exception as e:
                    st.error(f"Error loading alternative BDAP datasets: {str(e)}")
                    st.info("Please try using the primary API option or try again later.")
            
            # Only show the dataset search and selection if datasets are loaded
            if datasets_loaded:
                # Filter datasets related to local budgets first to provide better defaults
                suggested_datasets = [d for d in all_datasets if "bilanci" in d.lower() or "enti local" in d.lower()]
                
                # Search box for filtering datasets
                search_term = st.text_input("Search datasets", "bilanci enti locali")
                
                if search_term:
                    filtered_datasets = [d for d in all_datasets if search_term.lower() in d.lower()]
                else:
                    filtered_datasets = suggested_datasets if suggested_datasets else all_datasets[:50]  # Limit to first 50 by default
                
                # Show count of matching datasets
                st.caption(f"Found {len(filtered_datasets)} matching datasets out of {len(all_datasets)} total")
                
                # Select a dataset
                selected_dataset = st.selectbox(
                    "Select a dataset", 
                    filtered_datasets, 
                    index=0 if filtered_datasets else None
                )
                
                if selected_dataset:
                    # Button to fetch dataset details
                    if st.button("Get Dataset Details"):
                        with st.spinner(f"Fetching details for dataset: {selected_dataset}"):
                            try:
                                # Get the details for the selected dataset
                                dataset_details = get_alt_dataset_details(selected_dataset)
                                
                                # Store the details in session state
                                st.session_state.alt_dataset_details = dataset_details
                                
                                # Show success message
                                st.success(f"Successfully fetched details for: {selected_dataset}")
                            except Exception as e:
                                st.error(f"Error fetching dataset details: {str(e)}")
                
            # If we have dataset details, show them
            if "alt_dataset_details" in st.session_state and st.session_state.alt_dataset_details:
                details = st.session_state.alt_dataset_details
                
                # Display dataset information
                st.subheader("Dataset Information")
                
                # Extract and show metadata
                if "metadata" in details:
                    metadata = details["metadata"]
                    st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                    st.write(f"**Description:** {metadata.get('description', 'N/A')}")
                    st.write(f"**Last Updated:** {metadata.get('last_modified', 'N/A')}")
                
                # Extract and show resources
                if "resources" in details:
                    resources = details["resources"]
                    st.subheader("Resources")
                    
                    # Filter for CSV resources
                    csv_resources = [r for r in resources if r.get("format", "").lower() == "csv"]
                    
                    if csv_resources:
                        selected_resource = st.selectbox(
                            "Select a CSV resource to visualize",
                            csv_resources,
                            format_func=lambda x: f"{x.get('name', 'Unnamed')} - {x.get('description', 'No description')}"
                        )
                        
                        if selected_resource:
                            # Button to download and visualize the resource
                            if st.button("Download and Visualize Data"):
                                with st.spinner("Downloading and processing data..."):
                                    try:
                                        # Get the URL and download the CSV
                                        url = selected_resource.get("url")
                                        if url:
                                            # Download the CSV data
                                            response = requests.get(url, timeout=30)
                                            response.raise_for_status()
                                            
                                            # Parse the CSV data
                                            data = pd.read_csv(io.StringIO(response.text), sep=None, engine='python')
                                            
                                            # Store the data in session state
                                            st.session_state.alt_data = {
                                                "data": data,
                                                "name": selected_resource.get("name", "Unknown")
                                            }
                                            
                                            # Show success message
                                            st.success(f"Successfully downloaded and processed data: {len(data)} rows")
                                        else:
                                            st.error("No URL found for the selected resource")
                                    except Exception as e:
                                        st.error(f"Error downloading or processing data: {str(e)}")
                    else:
                        st.warning("No CSV resources found in this dataset")
                
                # If we have data, offer visualization options
                if "alt_data" in st.session_state and st.session_state.alt_data:
                    data = st.session_state.alt_data["data"]
                    name = st.session_state.alt_data["name"]
                    
                    st.subheader(f"Visualization: {name}")
                    
                    # Show a preview of the data
                    st.subheader("Data Preview")
                    st.dataframe(data.head(10))
                    
                    # Download link for the full CSV
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download full CSV",
                        data=csv,
                        file_name=f"{name}.csv",
                        mime="text/csv"
                    )
                    
                    # Try to identify numeric columns for visualization
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    if numeric_cols:
                        st.subheader("Data Visualization")
                        
                        # Let user select columns for visualization
                        x_axis = st.selectbox("Select X-axis", data.columns.tolist())
                        y_axis = st.selectbox("Select Y-axis", numeric_cols if numeric_cols else data.columns.tolist())
                        
                        if x_axis and y_axis:
                            # Create visualization based on data types
                            chart_type = st.radio("Chart type", ["Bar", "Line"], horizontal=True)
                            
                            if chart_type == "Bar":
                                chart = alt.Chart(data).mark_bar().encode(
                                    x=x_axis,
                                    y=y_axis,
                                    tooltip=[x_axis, y_axis]
                                ).properties(title=f"{name} - {x_axis} vs {y_axis}").interactive()
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                chart = alt.Chart(data).mark_line().encode(
                                    x=x_axis,
                                    y=y_axis,
                                    tooltip=[x_axis, y_axis]
                                ).properties(title=f"{name} - {x_axis} vs {y_axis}").interactive()
                                st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("No numeric columns found for visualization. Try another dataset.")
        
        # Add alternate data source option (moved to the bottom for both API options)
        st.markdown("---")
        st.subheader("External Resources")
        st.caption("Having trouble with the BDAP API? Try one of these alternatives:")
        alt_source_col1, alt_source_col2, alt_source_col3 = st.columns(3)
        with alt_source_col1:
            st.markdown("[OpenBDAP Portal](https://openbdap.rgs.mef.gov.it/)")
        with alt_source_col2:
            st.markdown("[OpenBilanci](https://openbilanci.it/pages/confronti)")
        with alt_source_col3:
            st.markdown("[BDAP Documentation](https://bdap-opendata.rgs.mef.gov.it/SpodCkanApi/en/Home)")
        
        # Process form submission
        if submitted:
            try:
                with st.spinner("Fetching data from BDAP API..."):
                    # Store data in session state to persist between reruns
                    st.session_state.bdap_data = fetch_and_visualize_bdap_data(
                        regione, years_range, data_type, top_n
                    )
                
                # Visualize data if available
                if st.session_state.bdap_data:
                    st.success("Data fetched successfully!")
                    visualize_bdap_data(st.session_state.bdap_data, regione, data_type, top_n)
                else:
                    st.error("Unable to retrieve data from BDAP. The server might be temporarily unavailable.")
                    st.info("Try selecting a different region or a smaller year range.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("The BDAP API may be experiencing issues. Please try again later or use the alternative sources.")
        elif "bdap_data" in st.session_state and st.session_state.bdap_data:
            # Show previous visualization if available
            visualize_bdap_data(st.session_state.bdap_data, regione, data_type, top_n)
        else:
            st.info(ui["no_data"])
    
    # Tab 5: Data Formulator
    with tab_formulator:
        st.header("Data Formulator Integration")
        st.markdown("""
        [Data Formulator](https://github.com/microsoft/data-formulator) is a powerful tool from Microsoft Research 
        that uses AI to transform data and create rich visualizations.
        
        You can use it to create interactive visualizations with your financial data from BDAP or any other data source.
        """)
        
        # Check if Data Formulator is installed
        import importlib.util
        data_formulator_installed = importlib.util.find_spec("data_formulator") is not None
        
        if data_formulator_installed:
            st.success("✅ Data Formulator is installed. You can launch it now.")
            
            # Option to set custom port
            col1, col2 = st.columns(2)
            with col1:
                port = st.number_input("Port (default: 5000)", min_value=1000, max_value=9999, value=5000)
            with col2:
                custom_args = st.text_input("Additional arguments", value="", placeholder="--debug")
            
            # Launch button
            if st.button("🚀 Launch Data Formulator", type="primary"):
                try:
                    # Alternative approach: create a command file and run it
                    import os
                    import sys
                    
                    # Create a launcher script
                    launcher_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch_data_formulator.py")
                    
                    # Prepare the custom args code if needed
                    custom_args_code = ""
                    if custom_args:
                        custom_args_code = f"custom_args = '{custom_args}'\ncmd.extend(custom_args.split())"
                    
                    with open(launcher_path, "w") as f:
                        f.write(f'''
# Data Formulator launcher script
import os
import sys
import subprocess

# Run Data Formulator
cmd = [sys.executable, "-m", "data_formulator", "--port", "{port}"]
{custom_args_code}

try:
    print("Starting Data Formulator...")
    print(f"Command: {{' '.join(cmd)}}")
    process = subprocess.Popen(cmd)
    print(f"Data Formulator started with PID: {{process.pid}}")
    print(f"Access at: http://localhost:{port}")
except Exception as e:
    print(f"Error launching Data Formulator: {{str(e)}}")
''')
                    
                    # Execute the launcher in a separate process
                    import subprocess
                    subprocess.Popen([sys.executable, launcher_path], 
                                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                    
                    st.success(f"Data Formulator launch initiated! It should be available at http://localhost:{port} in a few seconds.")
                    st.markdown(f"""
                    <div style="margin-top: 20px">
                        <a href='http://localhost:{port}' target='_blank'>
                            <button style='background-color:#4CAF50;color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;'>
                            Open Data Formulator in New Tab</button>
                        </a>
                    </div>
                    <div style="margin-top: 10px">
                        <p>If the page doesn't load immediately, wait a few seconds and try again. A separate console window should appear showing the launch status.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Also provide direct command
                    with st.expander("Alternative Manual Launch"):
                        cmd = f"python -m data_formulator --port {port} {custom_args}"
                        st.code(cmd)
                        st.markdown("You can run this command in a terminal if the automatic launch doesn't work.")
                    
                except Exception as e:
                    st.error(f"Error launching Data Formulator: {str(e)}")
        else:
            st.warning("⚠️ Data Formulator is not installed in this environment.")
            
            # Installation instructions
            with st.expander("Installation Instructions"):
                st.markdown("""
                ### Install Data Formulator with pip
                
                ```bash
                pip install data_formulator
                ```
                
                ### Alternative: Clone the repository
                
                ```bash
                git clone https://github.com/microsoft/data-formulator.git
                cd data-formulator
                pip install -e .
                ```
                
                After installation, restart this application to enable the Data Formulator launcher.
                """)
            
            # Install with pip button
            if st.button("📦 Install Data Formulator"):
                try:
                    import subprocess
                    with st.spinner("Installing Data Formulator..."):
                        result = subprocess.run(
                            ["pip", "install", "data_formulator"], 
                            capture_output=True, 
                            text=True,
                            check=True
                        )
                        st.success("Data Formulator installed successfully! Please restart this app to use it.")
                        st.code(result.stdout)
                except subprocess.CalledProcessError as e:
                    st.error(f"Error installing Data Formulator: {e.stderr}")
        
        # Resources section
        st.markdown("---")
        st.subheader("Resources & Documentation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("[📚 GitHub Repository](https://github.com/microsoft/data-formulator)")
            st.markdown("[📝 Research Paper](https://arxiv.org/abs/2408.16119)")
        
        with col2:
            st.markdown("[🎥 Demo Video](https://youtu.be/eGR2MG6l-o0)")
            st.markdown("[📖 Documentation](https://github.com/microsoft/data-formulator/blob/main/DEVELOPMENT.md)")
            
        # Use case integration
        st.markdown("---")
        st.subheader("BDAP Data Integration")
        st.markdown("""
        ### Using BDAP Data with Data Formulator
        
        1. Download CSV data from the Financial Data tab
        2. Launch Data Formulator
        3. Upload the CSV file to Data Formulator
        4. Use Data Formulator's AI-powered tools to create visualizations
        
        This allows you to create advanced visualizations of Italian financial data beyond what's available in this app!
        """)
        
        # Latest features
        with st.expander("Latest Features (April 2025)"):
            st.markdown("""
            ### Data Formulator 0.2: working with large data 📦📦📦
            
            - Upload large data file to the local database (powered by DuckDB)
            - Use drag-and-drop to specify charts, with dynamic data fetching
            - Work with AI agents to generate SQL queries and transform data
            - Anchor results, follow up, create branches, join tables for deeper analysis
            
            ### Supported Models
            
            - OpenAI, Azure, Ollama, and Anthropic models (via LiteLLM)
            - Models with strong code generation recommended (gpt-4o, claude-3-5-sonnet, etc.)
            """)
    
    # Tab 6: Settings
    with tab_settings:
        # RAG Settings
        rag_settings = display_rag_settings(language)
        
    # Add chat input outside of all containers
    if selected_question:
        # Set the selected question as the user input
        user_input = selected_question
    else:
        # Get user input
        user_input = st.chat_input(ui["chat_placeholder"])
    
    # Process user input
    if user_input:
        # Add timestamp to message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get response
        with st.spinner(ui["processing"]):
            # Get RAG settings
            top_k = 5
            if "similarity_top_k" in st.session_state:
                top_k = st.session_state.similarity_top_k
            
            # Get relevant context
            context, sources = get_relevant_context(user_input, top_k=top_k)
            
            # Add web search results if enabled
            web_results = []
            if enable_web_search:
                web_results = search_web(user_input, max_results=3, language=language)
                if web_results:
                    web_context = "\n\n"
                    web_context += f"{ui['web_results']}:\n" if language == "Italiano" else "Web search results:\n"
                    for i, result in enumerate(web_results):
                        web_context += f"{i+1}. {result['title']}: {result['body']}\n"
                    context += web_context
            
            # Get API source from session state
            api_source = st.session_state.get("api_source", "groq")
            
            # Generate response using selected model and API source
            response = generate_response(user_input, context, sources, model_name, language, api_source)
            
            # Generate audio if enabled
            audio_data = None
            if enable_tts:
                audio_data = generate_audio(response, language)
            
            # Add assistant response to chat history with timestamp
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "audio": audio_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Reset feedback for new response
            st.session_state.feedback = None
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                
                # Play audio if available
                if audio_data and enable_tts:
                    audio_bytes = base64.b64decode(audio_data)
                    st.audio(audio_bytes, format="audio/mp3")
            
            # Display thinking process if enabled
            if show_thinking:
                with st.expander(ui["reasoning"], expanded=True):
                    st.markdown(f"### {ui['context']}")
                    st.text(context if context else ui["no_context"])
                    
                    if sources:
                        st.markdown(f"### {ui['sources']}")
                        for source in sources:
                            st.markdown(f"- {source}")
                    
                    if web_results:
                        st.markdown(f"### {ui['web_results']}")
                        for i, result in enumerate(web_results):
                            st.markdown(f"{i+1}. [{result['title']}]({result['href']})")
                            st.markdown(f"   {result['body']}")
            
        # Refresh the page to show feedback and post-response options
        st.rerun()

if __name__ == "__main__":
    main()
