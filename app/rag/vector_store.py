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

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the vector store"""
        with self._lock:
            try:
                if document_id not in self.document_id_to_chunks:
                    logger.warning(f"Document {document_id} not found in vector store")
                    return False

                chunk_indices = self.document_id_to_chunks[document_id]

                # Remove chunk metadata and update mappings
                chunks_to_remove = set(chunk_indices)
                new_chunks_metadata = []
                new_chunk_id_to_index = {}
                new_document_id_to_chunks = {}

                # Rebuild metadata without deleted chunks
                for old_idx, chunk_metadata in enumerate(self.chunks_metadata):
                    if old_idx not in chunks_to_remove:
                        new_idx = len(new_chunks_metadata)
                        new_chunks_metadata.append(chunk_metadata)
                        new_chunk_id_to_index[chunk_metadata.chunk_id] = new_idx

                        # Update document mappings
                        doc_id = chunk_metadata.document_id
                        if doc_id not in new_document_id_to_chunks:
                            new_document_id_to_chunks[doc_id] = []
                        new_document_id_to_chunks[doc_id].append(new_idx)

                # Update internal state
                self.chunks_metadata = new_chunks_metadata
                self.chunk_id_to_index = new_chunk_id_to_index
                self.document_id_to_chunks = new_document_id_to_chunks

                # Rebuild FAISS index (expensive but necessary for deletion)
                if self.chunks_metadata:
                    logger.info("Rebuilding FAISS index after deletion...")
                    chunk_texts = [chunk.content for chunk in self.chunks_metadata]
                    embeddings = self.embeddings_manager.encode_batch(chunk_texts, normalize=True)

                    self.index = self._initialize_index(self.embedding_dim)
                    self.index.add(embeddings.astype(np.float32))
                else:
                    # Empty store
                    self.index = None
                    self.embedding_dim = None

                # Update statistics
                self.stats["total_chunks"] = len(self.chunks_metadata)
                self.stats["total_documents"] = len(self.document_id_to_chunks)
                self.stats["last_updated"] = datetime.now().isoformat()
                self._update_index_size()

                logger.info(f"Deleted document {document_id} and {len(chunk_indices)} chunks")
                return True

            except Exception as e:
                logger.error(f"Failed to delete document {document_id}: {e}")
                return False

    def save_index(self) -> bool:
        """Save FAISS index and metadata to disk"""
        with self._lock:
            try:
                # Save FAISS index
                if self.index is not None:
                    faiss.write_index(self.index, str(self.index_path))
                    logger.info(f"Saved FAISS index to {self.index_path}")

                # Save metadata
                metadata_data = {
                    "chunks": [chunk.to_dict() for chunk in self.chunks_metadata],
                    "chunk_id_to_index": self.chunk_id_to_index,
                    "document_id_to_chunks": self.document_id_to_chunks,
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type
                }

                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_data, f, indent=2, ensure_ascii=False)

                # Save statistics
                with open(self.stats_path, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, indent=2)

                logger.info(f"Saved vector store metadata to {self.metadata_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")
                return False

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        with self._lock:
            try:
                # Check if files exist
                if not self.metadata_path.exists():
                    logger.info("No existing vector store metadata found")
                    return False

                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata_data = json.load(f)

                # Restore chunks metadata
                self.chunks_metadata = [
                    ChunkMetadata.from_dict(chunk_data)
                    for chunk_data in metadata_data.get("chunks", [])
                ]
                self.chunk_id_to_index = metadata_data.get("chunk_id_to_index", {})
                self.document_id_to_chunks = {
                    doc_id: indices for doc_id, indices in metadata_data.get("document_id_to_chunks", {}).items()
                }
                self.embedding_dim = metadata_data.get("embedding_dim")
                self.index_type = metadata_data.get("index_type", "IndexFlatIP")

                # Load FAISS index
                if self.index_path.exists() and self.chunks_metadata:
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(f"Loaded FAISS index from {self.index_path}")
                else:
                    logger.info("No FAISS index file found or no chunks in metadata")

                # Load statistics
                if self.stats_path.exists():
                    with open(self.stats_path, 'r', encoding='utf-8') as f:
                        self.stats.update(json.load(f))
                else:
                    # Calculate statistics
                    self.stats["total_chunks"] = len(self.chunks_metadata)
                    self.stats["total_documents"] = len(self.document_id_to_chunks)
                    self.stats["last_updated"] = datetime.now().isoformat()

                self._update_index_size()

                logger.info(f"Loaded vector store with {len(self.chunks_metadata)} chunks from {len(self.document_id_to_chunks)} documents")
                return True

            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                # Reset to empty state
                self.index = None
                self.chunks_metadata = []
                self.chunk_id_to_index = {}
                self.document_id_to_chunks = {}
                self.embedding_dim = None
                return False

    def clear(self) -> bool:
        """Clear all data from the vector store"""
        with self._lock:
            try:
                # Clear in-memory data
                self.index = None
                self.chunks_metadata = []
                self.chunk_id_to_index = {}
                self.document_id_to_chunks = {}
                self.embedding_dim = None

                # Reset statistics
                self.stats = {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "last_updated": datetime.now().isoformat(),
                    "index_size_mb": 0.0
                }

                # Remove files
                if self.index_path.exists():
                    self.index_path.unlink()
                if self.metadata_path.exists():
                    self.metadata_path.unlink()
                if self.stats_path.exists():
                    self.stats_path.unlink()

                logger.info("Cleared vector store")
                return True

            except Exception as e:
                logger.error(f"Failed to clear vector store: {e}")
                return False

    def _update_index_size(self):
        """Update index size in statistics"""
        try:
            if self.index_path.exists():
                size_bytes = self.index_path.stat().st_size
                self.stats["index_size_mb"] = size_bytes / (1024 * 1024)
            else:
                self.stats["index_size_mb"] = 0.0
        except Exception:
            self.stats["index_size_mb"] = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        with self._lock:
            self._update_index_size()
            return self.stats.copy()

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store"""
        with self._lock:
            documents = []
            for doc_id, chunk_indices in self.document_id_to_chunks.items():
                if chunk_indices:
                    # Get first chunk for document info
                    first_chunk = self.chunks_metadata[chunk_indices[0]]
                    documents.append({
                        "document_id": doc_id,
                        "chunk_count": len(chunk_indices),
                        "created_at": first_chunk.created_at.isoformat(),
                        "sample_content": first_chunk.content[:200] + "..." if len(first_chunk.content) > 200 else first_chunk.content
                    })

            # Sort by creation time (newest first)
            documents.sort(key=lambda x: x["created_at"], reverse=True)
            return documents

    def get_similar_chunks_for_rag(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_total_length: Optional[int] = None
    ) -> Tuple[List[SearchResult], str]:
        """
        Get similar chunks formatted for RAG context
        Returns (search_results, formatted_context)
        """
        max_chunks = max_chunks or settings.rag.max_chunks_per_query
        similarity_threshold = similarity_threshold or settings.rag.similarity_threshold
        max_total_length = max_total_length or settings.rag.max_context_length

        # Search for similar chunks
        results = self.search(
            query=query,
            k=max_chunks,
            similarity_threshold=similarity_threshold
        )

        if not results:
            return [], ""

        # Format context for RAG
        context_parts = []
        total_length = 0

        for result in results:
            chunk_text = result.chunk_metadata.content

            # Add source reference if enabled
            if settings.rag.include_source_references:
                source_ref = f"[Document: {result.chunk_metadata.document_id}"
                if result.chunk_metadata.page_number:
                    source_ref += f", Page: {result.chunk_metadata.page_number}"
                source_ref += f", Relevance: {result.similarity_score:.2f}]"
                chunk_text = f"{source_ref}\n{chunk_text}"

            # Check if adding this chunk would exceed max length
            if max_total_length and total_length + len(chunk_text) > max_total_length:
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        formatted_context = "\n\n---\n\n".join(context_parts)
        return results[:len(context_parts)], formatted_context


# Global vector store instance
vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    """Get the global vector store instance"""
    return vector_store
