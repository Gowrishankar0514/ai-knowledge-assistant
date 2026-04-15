from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from src.embeddings.vector_store import VectorStoreManager
from src.retrieval.reranker import RankBM25Reranker
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedRetriever:
    def __init__(self, vector_manager: VectorStoreManager, top_k=4):
        self.base_retriever = vector_manager.get_retriever(k=top_k * 2) # Fetch extra for re-ranking
        logger.info("Initializing baseline Dense Retriever from Vector Store.")

        # Hybrid/re-ranking approach using custom BM25 document compressor
        self.compressor = RankBM25Reranker(top_k=top_k)
        
        if self.base_retriever:
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, 
                base_retriever=self.base_retriever
            )
        else:
            self.compression_retriever = None

    def get_context(self, query: str):
        """Retrieves and re-ranks documents based on query."""
        if not self.compression_retriever:
            logger.error("Compression retriever not configured.")
            return []
            
        logger.info(f"Retrieving context for query: '{query}'")
        try:
            docs = self.compression_retriever.invoke(query)
            # Remove duplicated or highly similar chunks from context window if any
            unique_docs = self._filter_duplicates(docs)
            logger.info(f"Retrieved {len(unique_docs)} relevant context chunks.")
            return unique_docs
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            return []

    def _filter_duplicates(self, docs):
        """Basic filtering to remove duplicate contexts based on exact content match."""
        seen = set()
        filtered = []
        for doc in docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                filtered.append(doc)
        return filtered
