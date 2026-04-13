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

# Inject Custom CSS for Premium UI
def apply_custom_css():
    st.markdown("""
        <style>
        /* Main background tweaks */
        .stApp {
            background: radial-gradient(circle at top right, #111424, #0A0B10);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(21, 23, 32, 0.8) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Buttons glassmorphism */
        .stButton>button {
            border-radius: 8px;
            background: linear-gradient(90deg, #00A3FF, #00E5FF);
            color: #0A0B10;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
            transform: translateY(-2px);
            color: #ffffff;
        }

        /* Inputs */
        .stTextInput>div>div>input {
            border-radius: 8px;
            background-color: rgba(255,255,255, 0.05);
            border: 1px solid rgba(255,255,255, 0.1);
        }
        
        /* Chat bubble overrides */
        [data-testid="stChatMessage"] {
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        /* Title styling */
        h1 {
            background: -webkit-linear-gradient(45deg, #00E5FF, #00A3FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        </style>
    """, unsafe_allow_html=True)


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = ChatMemoryManager()
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager()
if "generator" not in st.session_state:
    st.session_state.generator = None
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = "Baseline Q&A"

def init_system(persona: str = None):
    p = persona or st.session_state.selected_persona
    st.session_state.vector_manager.load_index()
    if st.session_state.vector_manager.vector_store:
        retriever = AdvancedRetriever(st.session_state.vector_manager)
        st.session_state.generator = RAGGenerator(retriever, st.session_state.memory_manager, persona=p)
    else:
        st.session_state.generator = RAGGenerator(None, st.session_state.memory_manager, persona=p)

st.set_page_config(page_title="Cognitive Engine", layout="wide", initial_sidebar_state="expanded")
apply_custom_css()

# Header
st.title("Cognitive Engine AI")
st.markdown("### Enterprise Knowledge Assistant & Innovation Platform")

# Sidebar Configuration
with st.sidebar:
    st.header("🧠 Cognitive Settings")
    new_persona = st.selectbox(
        "Select AI Persona (Thinking Mode):",
        ["Baseline Q&A", "First Principles", "Lateral Ideation", "Devil's Advocate"],
        index=["Baseline Q&A", "First Principles", "Lateral Ideation", "Devil's Advocate"].index(st.session_state.selected_persona)
    )
    
    # If persona changed, update it
    if new_persona != st.session_state.selected_persona:
        st.session_state.selected_persona = new_persona
        if st.session_state.generator:
            st.session_state.generator.set_persona(new_persona)
        st.rerun()

    st.markdown("---")
    st.header("📥 Data Ingestion")
    
    # URL Ingestion
    st.markdown("###### Web Scraper")
    url_input = st.text_input("Enter URL:")
    if st.button("Extract & Index URL") and url_input:
        with st.spinner("Crawling structure..."):
            scraper = WebScraper(base_url=url_input, max_depth=1, max_pages=5)
            scraper.scrape()
            docs = scraper.get_documents()
            chunker = SemanticChunker()
            chunked = chunker.chunk_documents(docs)
            if chunked:
                st.session_state.vector_manager.build_index(chunked)
                init_system()
                st.success(f"Indexed {len(chunked)} chunks!")
            else:
                st.warning("No readability found.")

    # Local Files Ingestion
    st.markdown("###### Local Directory")
    if st.button("Process Local Folders"):
        with st.spinner("Analyzing `/data` folder..."):
            loader = DocumentLoader(Config.DATA_DIR)
            docs = loader.load_documents()
            chunker = SemanticChunker()
            chunked = chunker.chunk_documents(docs)
            if chunked:
                st.session_state.vector_manager.build_index(chunked)
                init_system()
                st.success(f"Indexed {len(chunked)} chunks!")
            else:
                st.warning(f"No valid files in `{Config.DATA_DIR}`.")
                
    st.markdown("---")
    if st.button("🗑️ Clear Memory"):
        st.session_state.memory_manager.clear()
        st.session_state.chat_history = []
        st.success("Session reset.")

# Check if generator isn't loaded yet
if st.session_state.generator is None:
    init_system(st.session_state.selected_persona)

# Chat Interface
st.markdown(f"**Current Mode:** `{st.session_state.selected_persona}`")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Cited Sources"):
                for src in message["sources"]:
                    st.caption(f"📄 {src}")

# User input
if user_question := st.chat_input("Engage with the Cognitive Engine..."):
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing..."):
            if not st.session_state.generator or not st.session_state.generator.qa_chain:
                response_text = "I require raw data to function. Please provide a URL or local files inside the ingestion panel."
                sources_list = []
            else:
                res = st.session_state.generator.query(user_question)
                response_text = res.get("answer", "Processing error.")
                docs = res.get("source_documents", [])
                sources_list = list(set([d.metadata.get("source", "Unknown") for d in docs]))
            
            st.markdown(response_text)
            if sources_list:
                with st.expander("View Cited Sources"):
                    for src in sources_list:
                        st.caption(f"📄 {src}")
                        
    # Add AI response to chat history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources_list
    })
