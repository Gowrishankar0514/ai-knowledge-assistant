# AI Knowledge Assistant (Enterprise-Grade RAG)

A complete end-to-end Retrieval-Augmented Generation (RAG) system inspired by enterprise AI platforms like CustomGPT.

## Features
- **Multi-Source Ingestion**: Recursively scrapes websites, extracts text from PDFs, and reads plain text documents.
- **Intelligent Chunking**: Semantic structure-based text chunking (headings, paragraphs) rather than fixed-size splitting.
- **Vector Search**: Embeds and indexes documents using HuggingFace and FAISS/ChromaDB.
- **Advanced Retrieval**: Performs top-K semantic search, re-ranking, and duplicate context removal.
- **Grounded LLM**: Integrates with OpenAI/HuggingFace to provide grounded, hallucination-free answers with source citations.
- **Conversational Memory**: Follow-up questions enabled through memory tracking.
- **Interactive UI**: Streamlit-based web interface.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
3. Run the UI:
   ```bash
   streamlit run app.py
   ```
