"""
Vector store module for RAG functionality
Handles vector similarity search using FAISS with persistent storage
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
import threading
import time

# Vector similarity search
import faiss

from app.core.config import settings
from app.rag.document_processor import DocumentChunk, Document
from app.rag.embeddings import get_embeddings_manager

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata for a chunk stored in the vector database"""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        chunk_metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.page_number = page_number
        self.start_char = start_char
        self.end_char = end_char
        self.chunk_metadata = chunk_metadata or {}
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_metadata": self.chunk_metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create from dictionary"""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            content=data["content"],
            page_number=data.get("page_number"),
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            chunk_metadata=data.get("chunk_metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None
        )

    @classmethod
    def from_document_chunk(cls, chunk: DocumentChunk) -> "ChunkMetadata":
        """Create from DocumentChunk"""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            chunk_metadata=chunk.metadata,
            created_at=chunk.created_at
        )


class SearchResult:
    """Result from vector similarity search"""

    def __init__(
        self,
        chunk_metadata: ChunkMetadata,
        similarity_score: float,
        rank: int,
        embedding: Optional[np.ndarray] = None
    ):
        self.chunk_metadata = chunk_metadata
        self.similarity_score = similarity_score
        self.rank = rank
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunk_id": self.chunk_metadata.chunk_id,
            "document_id": self.chunk_metadata.document_id,
            "content": self.chunk_metadata.content,
            "page_number": self.chunk_metadata.page_number,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "metadata": self.chunk_metadata.chunk_metadata
        }


class VectorStore:
    """Vector store for document chunks using FAISS"""

    def __init__(self, vector_db_dir: Optional[str] = None):
        self.vector_db_dir = Path(vector_db_dir or settings.rag.vector_db_dir)
        self.vector_db_dir.mkdir(exist_ok=True, parents=True)

        # FAISS index and metadata
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[ChunkMetadata] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        self.document_id_to_chunks: Dict[str, List[int]] = {}

        # Embeddings manager
        self.embeddings_manager = get_embeddings_manager()

        # Index configuration
        self.embedding_dim: Optional[int] = None
        self.index_type = "IndexFlatIP"  # Inner Product (cosine similarity for normalized vectors)

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_chunks": 0,
            "total_documents": 0,
            "last_updated": None,
            "index_size_mb": 0.0
        }

        # File paths
        self.index_path = self.vector_db_dir / "faiss_index.bin"
        self.metadata_path = self.vector_db_dir / "metadata.json"
        self.stats_path = self.vector_db_dir / "stats.json"

        logger.info(f"VectorStore initialized with directory: {self.vector_db_dir}")

        # Try to load existing index
        self.load_index()

    def _ensure_embeddings_loaded(self) -> bool:
        """Ensure embeddings model is loaded"""
        if not self.embeddings_manager.is_loaded:
            logger.info("Loading embeddings model for vector store...")
            return self.embeddings_manager.load_model()
        return True

    def _initialize_index(self, embedding_dim: int) -> faiss.Index:
        """Initialize FAISS index with given dimension"""
        if self.index_type == "IndexFlatIP":
            # Inner Product index (good for cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)
        elif self.index_type == "IndexFlatL2":
            # L2 distance index
            index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Default to Inner Product
            index = faiss.IndexFlatIP(embedding_dim)

        logger.info(f"Initialized FAISS index: {self.index_type} with dimension {embedding_dim}")
        return index

    def add_document(self, document: Document, batch_size: int = 32) -> bool:
        """Add a document and its chunks to the vector store"""
        with self._lock:
            try:
                if not self._ensure_embeddings_loaded():
                    raise RuntimeError("Failed to load embeddings model")

                if not document.chunks:
                    logger.warning(f"Document {document.document_id} has no chunks to add")
                    return False

                # Get embedding dimension if not set
                if self.embedding_dim is None:
                    self.embedding_dim = self.embeddings_manager.embedding_dim
                    if self.embedding_dim is None:
                        # Get dimension by encoding a sample text
                        sample_embedding = self.embeddings_manager.encode_text("sample text")
                        self.embedding_dim = sample_embedding.shape[0]

                # Initialize index if needed
                if self.index is None:
                    self.index = self._initialize_index(self.embedding_dim)

                # Check if document already exists
                if document.document_id in self.document_id_to_chunks:
                    logger.warning(f"Document {document.document_id} already exists in vector store")
                    return False

                # Extract chunk texts for batch embedding
                chunk_texts = [chunk.content for chunk in document.chunks]

                # Generate embeddings in batches
                logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
                embeddings = self.embeddings_manager.encode_batch(
                    chunk_texts,
                    normalize=True,
                    batch_size=batch_size,
                    show_progress=len(chunk_texts) > 50
                )

                # Prepare chunk metadata
                chunk_indices = []
                for i, chunk in enumerate(document.chunks):
                    chunk_metadata = ChunkMetadata.from_document_chunk(chunk)

                    # Add to metadata storage
                    chunk_index = len(self.chunks_metadata)
                    self.chunks_metadata.append(chunk_metadata)
                    self.chunk_id_to_index[chunk.chunk_id] = chunk_index
                    chunk_indices.append(chunk_index)

                # Add document to chunks mapping
                self.document_id_to_chunks[document.document_id] = chunk_indices

                # Add embeddings to FAISS index
                self.index.add(embeddings.astype(np.float32))

                # Update statistics
                self.stats["total_chunks"] += len(document.chunks)
                if document.document_id not in self.document_id_to_chunks or len(self.document_id_to_chunks[document.document_id]) == len(chunk_indices):
                    self.stats["total_documents"] += 1
                self.stats["last_updated"] = datetime.now().isoformat()
                self._update_index_size()

                logger.info(f"Added document {document.document_id} with {len(document.chunks)} chunks to vector store")
                return True

            except Exception as e:
                logger.error(f"Failed to add document {document.document_id}: {e}")
                return False

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks using vector similarity"""
        with self._lock:
            try:
                if not self._ensure_embeddings_loaded():
                    raise RuntimeError("Failed to load embeddings model")

                if self.index is None or len(self.chunks_metadata) == 0:
                    logger.warning("Vector store is empty")
                    return []

                # Use default values from settings
                k = k or settings.rag.max_chunks_per_query
                similarity_threshold = similarity_threshold or settings.rag.similarity_threshold

                # Generate query embedding
                query_embedding = self.embeddings_manager.encode_text(query, normalize=True)
                query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

                # Search FAISS index
                search_k = min(k * 2, len(self.chunks_metadata))  # Get more results for filtering
                scores, indices = self.index.search(query_embedding, search_k)

                # Process results
                results = []
                for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx == -1:  # FAISS returns -1 for invalid indices
                        continue

                    if idx >= len(self.chunks_metadata):
                        logger.warning(f"Invalid chunk index: {idx}")
                        continue

                    chunk_metadata = self.chunks_metadata[idx]

                    # Filter by document IDs if specified
                    if document_ids and chunk_metadata.document_id not in document_ids:
                        continue

                    # Filter by similarity threshold
                    if score < similarity_threshold:
                        continue

                    result = SearchResult(
                        chunk_metadata=chunk_metadata,
                        similarity_score=float(score),
                        rank=len(results) + 1
                    )

                    results.append(result)

                    # Stop when we have enough results
                    if len(results) >= k:
                        break

                logger.info(f"Found {len(results)} relevant chunks for query (threshold: {similarity_threshold})")
                return results

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get chunk metadata by chunk ID"""
        with self._lock:
            if chunk_id in self.chunk_id_to_index:
                idx = self.chunk_id_to_index[chunk_id]
                if idx < len(self.chunks_metadata):
                    return self.chunks_metadata[idx]
            return None

    def get_chunks_by_document_id(self, document_id: str) -> List[ChunkMetadata]:
        """Get all chunks for a document"""
        with self._lock:
            if document_id not in self.document_id_to_chunks:
                return []

            chunk_indices = self.document_id_to_chunks[document_id]
            chunks = []

            for idx in chunk_indices:
                if idx < len(self.chunks_metadata):
                    chunks.append(self.chunks_metadata[idx])

            return chunks
