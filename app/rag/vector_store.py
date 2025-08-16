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


        # Statistics
        self.stats = {
            "total_chunks": 0,
            "total_documents": 0,
            "last_updated": None,
            "index_size_mb": 0.0
        }
