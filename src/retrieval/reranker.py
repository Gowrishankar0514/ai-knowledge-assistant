from typing import Sequence
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RankBM25Reranker(BaseDocumentCompressor):
    """
    Custom Document Compressor that re-ranks retrieved dense chunks 
    using BM25 algorithm (Hybrid Search simulation) to eliminate irrelevant context.
    """
    top_k: int = 4

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        
        if not documents:
            return documents

        # Tokenize documents and query
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in documents]
        tokenized_query = query.lower().split(" ")
        
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Pair scores with documents
        scored_docs = list(zip(documents, scores))
        # Sort desc by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        filtered_docs = [doc for doc, score in scored_docs[:self.top_k]]
        return filtered_docs
