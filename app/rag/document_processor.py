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

# Document parsing libraries
import PyPDF2
import fitz  # PyMuPDF for better PDF handling
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of text from a document"""

    def __init__(
        self,
        content: str,
        chunk_id: str,
        document_id: str,
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page_number = page_number
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}
        self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create chunk from dictionary"""
        chunk = cls(
            content=data["content"],
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            page_number=data.get("page_number"),
            start_char=data.get("start_char"),
            end_char=data.get("end_char"),
            metadata=data.get("metadata", {})
        )
        if "created_at" in data:
            chunk.created_at = datetime.fromisoformat(data["created_at"])
        return chunk


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
