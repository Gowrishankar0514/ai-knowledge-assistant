import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.store_type = Config.VECTOR_STORE_TYPE.lower()
        self.db_dir = Config.VECTOR_DB_DIR
        self.vector_store = None

        # Change db dir to match chroma if it's chroma
        if self.store_type == 'chroma':
            self.db_dir = os.path.join(os.path.dirname(Config.VECTOR_DB_DIR), "chroma_db")

    def build_index(self, documents: list[Document]):
        """Builds a new vector index from a list of Langchain Document objects."""
        if not documents:
            logger.warning("No documents provided to build the index.")
            return

        logger.info(f"Building {self.store_type} vector store with {len(documents)} chunks...")
        
        try:
            if self.store_type == 'faiss':
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
                self.vector_store.save_local(self.db_dir)
                logger.info(f"FAISS index built and saved to {self.db_dir}")
                
            elif self.store_type == 'chroma':
                self.vector_store = Chroma.from_documents(
                    documents, 
                    self.embedding_model, 
                    persist_directory=self.db_dir
                )
                logger.info(f"Chroma index built and persisted to {self.db_dir}")
            else:
                logger.error(f"Unsupported VECTOR_STORE_TYPE: {self.store_type}")
                
        except Exception as e:
            logger.error(f"Error building vector index: {e}")

    def load_index(self):
        """Loads an existing vector index from disk."""
        logger.info(f"Loading {self.store_type} vector store from {self.db_dir}")
        try:
            if self.store_type == 'faiss':
                if os.path.exists(os.path.join(self.db_dir, "index.faiss")):
                    self.vector_store = FAISS.load_local(
                        self.db_dir, 
                        self.embedding_model, 
                        allow_dangerous_deserialization=True
                    )
                else:
                    logger.warning(f"No FAISS index found at {self.db_dir}")

            elif self.store_type == 'chroma':
                self.vector_store = Chroma(
                    persist_directory=self.db_dir,
                    embedding_function=self.embedding_model
                )
            else:
                logger.error(f"Unsupported VECTOR_STORE_TYPE: {self.store_type}")
                
        except Exception as e:
            logger.error(f"Error loading vector index: {e}")

    def get_retriever(self, k=4):
        """Returns a configured retriever from the loaded vector store."""
        if not self.vector_store:
            logger.warning("Vector store is not loaded. Cannot create retriever.")
            return None
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})
