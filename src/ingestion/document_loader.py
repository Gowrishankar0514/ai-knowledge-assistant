import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.utils.logger import get_logger
from src.ingestion.data_cleaner import clean_text

logger = get_logger(__name__)

class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents = []

    def load_documents(self):
        """Loads all supported documents from the data directory."""
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist.")
            return []

        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            
            if os.path.isdir(file_path):
                continue

            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        cleaned = clean_text(doc.page_content)
                        if cleaned.strip():
                            self.documents.append({
                                "source": filename,
                                "content": cleaned,
                                "type": "pdf"
                            })
                    logger.info(f"Successfully loaded PDF: {filename}")
                
                elif filename.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    for doc in docs:
                        cleaned = clean_text(doc.page_content)
                        if cleaned.strip():
                            self.documents.append({
                                "source": filename,
                                "content": cleaned,
                                "type": "txt"
                            })
                    logger.info(f"Successfully loaded TXT: {filename}")
                    
                else:
                    logger.warning(f"Unsupported file format: {filename}")
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

        return self.documents
