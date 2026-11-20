# SemantiCheck 3D — Text Similarity & Duplicate Detection

An end-to-end NLP application that evaluates lexical and semantic similarity between pasted text, Notepad (`.txt`), Word (`.docx`), and `.pdf` documents.

## Features
- **Dual NLP Engines:** Compare classical **TF-IDF + Cosine Similarity** against deep **Sentence-BERT (`all-MiniLM-L6-v2`)** embeddings.
- **Universal File Support:** Paste raw sentences/paragraphs or upload `.txt`, `.docx`, and `.pdf` files directly.
- **3D Matte-Dark Web GUI:** Built with Streamlit and Plotly featuring 3D elevated cards and an interactive 3D vector embedding projection.

## Setup & Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset, train TF-IDF model, and compute evaluation metrics
python -m src.train_and_evaluate

# 3. Launch the 3D Streamlit Web App
streamlit run app.py
```
