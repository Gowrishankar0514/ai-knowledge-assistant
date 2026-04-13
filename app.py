import streamlit as st
import os

# Prevent PyTorch OpenMP conflict if it happens locally
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.config import Config
from src.ingestion.web_scraper import WebScraper
from src.ingestion.document_loader import DocumentLoader
from src.embeddings.chunker import SemanticChunker
from src.embeddings.vector_store import VectorStoreManager
from src.retrieval.retriever import AdvancedRetriever
from src.llm.memory import ChatMemoryManager
from src.llm.generator import RAGGenerator

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = ChatMemoryManager()
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()
if "generator" not in st.session_state:
    st.session_state.generator = None

def init_system():
    # Load existing index if present
    st.session_state.vector_manager.load_index()
    if st.session_state.vector_manager.vector_store:
        retriever = AdvancedRetriever(st.session_state.vector_manager)
        st.session_state.generator = RAGGenerator(retriever, st.session_state.memory_manager)
    else:
        # Initialize without vector store / retriever
        st.session_state.generator = RAGGenerator(None, st.session_state.memory_manager)

st.set_page_config(page_title="Enterprise RAG Chatbot", layout="wide")

st.title("Enterprise AI Knowledge Assistant")
st.markdown("Upload documents or scrape websites, then ask context-aware questions.")

# Sidebar for Ingestion
with st.sidebar:
    st.header("1. Data Ingestion")
    
    # URL Ingestion
    url_input = st.text_input("Enter Website URL to Scrape:")
    if st.button("Scrape Website") and url_input:
        with st.spinner("Scraping website..."):
            scraper = WebScraper(base_url=url_input, max_depth=1, max_pages=5)
            scraper.scrape()
            docs = scraper.get_documents()
            
            chunker = SemanticChunker()
            chunked = chunker.chunk_documents(docs)
            
            if chunked:
                st.session_state.vector_manager.build_index(chunked)
                init_system()
                st.success(f"Successfully scraped and indexed {len(chunked)} chunks!")
            else:
                st.warning("No readable text found at URL.")

    # Local Files Ingestion
    st.markdown("---")
    st.subheader("Process Local Files")
    st.write(f"Place `.pdf` and `.txt` files in `{Config.DATA_DIR}`")
    if st.button("Process Local Directory"):
        with st.spinner("Loading local documents..."):
            loader = DocumentLoader(Config.DATA_DIR)
            docs = loader.load_documents()
            
            chunker = SemanticChunker()
            chunked = chunker.chunk_documents(docs)
            
            if chunked:
                st.session_state.vector_manager.build_index(chunked)
                init_system()
                st.success(f"Successfully indexed {len(chunked)} chunks from local files!")
            else:
                st.warning("No valid documents found in data directory.")
                
    st.markdown("---")
    if st.button("Clear Conversation Memory"):
        st.session_state.memory_manager.clear()
        st.session_state.chat_history = []
        st.success("Memory cleared.")

# Check if generator isn't loaded yet
if st.session_state.generator is None:
    init_system()

# Main Chat Interface
st.header("2. Chat Interface")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# User input
if user_question := st.chat_input("Ask a question about the provided context..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not st.session_state.generator or not st.session_state.generator.qa_chain:
                response_text = "I don't have any knowledge base ingested. Please provide a URL or local files first."
                sources_list = []
            else:
                res = st.session_state.generator.query(user_question)
                response_text = res.get("answer", "Error generating response.")
                
                # Extract sources
                docs = res.get("source_documents", [])
                sources_list = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            
            st.markdown(response_text)
            if sources_list:
                with st.expander("Sources"):
                    for src in sources_list:
                        st.write(f"- {src}")
                        
    # Add AI response to chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_list
    })
