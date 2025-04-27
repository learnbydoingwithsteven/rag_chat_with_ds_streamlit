# Multilingual Financial Statement RAG Chatbot - Streamlit Version

This is the Streamlit implementation of the Multilingual Financial Statement RAG Chatbot for harmonized financial statements.

## Features

- **Multilingual Support**: Toggle between English and Italian interfaces
- **Vector-based RAG**: Find and present the most relevant information from your financial documents
- **Document Management**: Upload and process PDF documents
- **Vector Visualization**: Interactive visualization of document embeddings with multi-document selection and color coding
- **Financial Data Access**: Access Italian government financial datasets via:
  - Primary BDAP CKAN API
  - Alternative BDAP REST API
- **Data Formulator Integration**: Launch and use Data Formulator directly from the app

## Project Structure

```
2_chatbot_multilingual.py # Main Streamlit application
0_pdf_to_text.py          # Script to extract text from PDFs
1_text_to_vector_db.py    # Script to create vector embeddings
bilanci_vectors.db        # SQLite database with vector embeddings
bilanci_pdf/              # Directory for PDF documents
```

## Requirements

- Python 3.8+
- Streamlit
- Pandas, NumPy, Matplotlib, Plotly
- Scikit-learn
- SQLite3
- Requests
- Optional: data_formulator package

## Setup and Installation

1. Install the required Python packages:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn plotly
pip install requests sqlite3 python-dotenv
pip install data-formulator # Optional, for Data Formulator integration
```

2. Run the preprocessing scripts if needed:

```bash
# Convert PDFs to text
python 0_pdf_to_text.py

# Create vector embeddings database
python 1_text_to_vector_db.py
```

3. Start the Streamlit app:

```bash
streamlit run 2_chatbot_multilingual.py
```

The app will open in your default web browser.

## Using the Application

The application provides five main tabs:

1. **Chat**: Ask questions about harmonized financial statements in either English or Italian
2. **Documents**: Upload and manage documents for the vector database
3. **Vector Visualization**: Visualize document embeddings with interactive PCA and t-SNE plots
   - Select multiple documents with color coding
   - Explore document similarities visually
4. **Financial Data**: Access and visualize Italian government financial datasets
   - Switch between primary CKAN API and alternative REST API
   - Download and visualize CSV resources
5. **Data Formulator**: Launch Data Formulator for advanced financial data analysis

## Notes

- The Streamlit version has all functionality in a single Python file for simplicity
- Make sure the database file exists in the same directory as the script
- For best results, upload high-quality, text-searchable PDFs
- The multilingual interface supports both English and Italian
