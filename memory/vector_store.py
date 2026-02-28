# memory/vector_store.py
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from .ingestion import SentenceTransformerEmbeddings  # reuse the class
from .ingestion import Document

class QdrantVectorStore:
    """A simple wrapper around Qdrant for similarity search."""
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embedder: SentenceTransformerEmbeddings):
        self.qdrant = qdrant_client
        self.collection = collection_name
        self.embedder = embedder
   
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_vector = self.embedder.embed_query(query)
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k
        )
        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.payload.get("text", ""),
                metadata=r.payload
            ))
        return docs


class VectorStoreManager:
    """Caches vector store instances per collection."""
    def __init__(self, qdrant_client: QdrantClient, embedding_model: str):
        self.qdrant = qdrant_client
        self.embedder = SentenceTransformerEmbeddings(embedding_model)
        self._stores = {}

    def get_store(self, collection_name: str) -> QdrantVectorStore:
        if collection_name not in self._stores:
            self._stores[collection_name] = QdrantVectorStore(
                self.qdrant,
                collection_name,
                self.embedder
            )
        return self._stores[collection_name]