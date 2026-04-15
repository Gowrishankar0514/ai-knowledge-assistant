from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SemanticChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # We use RecursiveCharacterTextSplitter as it semantically breaks by paragraphs, then sentences
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: list) -> list:
        """
        Takes a list of dictionaries with 'source', 'content', 'type'.
        Returns a list of Langchain Document objects.
        """
        from langchain_core.documents import Document
        
        chunked_docs = []
        for doc in documents:
            if not doc.get('content'):
                continue
            try:
                chunks = self.splitter.split_text(doc['content'])
                for i, chunk in enumerate(chunks):
                    chunked_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": doc.get('source', 'unknown'),
                                "type": doc.get('type', 'unknown'),
                                "chunk_id": i
                            }
                        )
                    )
            except Exception as e:
                logger.error(f"Error chunking document {doc.get('source', 'Unknown')}: {e}")
                
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents.")
        return chunked_docs
