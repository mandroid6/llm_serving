"""
Document processing module for RAG functionality
Handles PDF/TXT parsing, text chunking, and metadata extraction
"""

import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Document:
    """Represents a processed document with metadata"""

    def __init__(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        content: str,
        file_size: int,
        file_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.document_id = document_id
        self.filename = filename
        self.file_type = file_type
        self.content = content
        self.file_size = file_size
        self.file_hash = file_hash
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.chunks: List[DocumentChunk] = []

    def add_chunk(self, chunk: DocumentChunk):
        """Add a chunk to this document"""
        self.chunks.append(chunk)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for storage"""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "content": self.content,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "chunk_count": len(self.chunks)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary"""
        doc = cls(
            document_id=data["document_id"],
            filename=data["filename"],
            file_type=data["file_type"],
            content=data["content"],
            file_size=data["file_size"],
            file_hash=data["file_hash"],
            metadata=data.get("metadata", {})
        )
        doc.created_at = datetime.fromisoformat(data["created_at"])
        doc.updated_at = datetime.fromisoformat(data["updated_at"])
        return doc
