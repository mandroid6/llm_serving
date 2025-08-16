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


class DocumentProcessor:
    """Main document processing class"""

    def __init__(self):
        self.documents_dir = Path(settings.rag.documents_dir)
        self.documents_dir.mkdir(exist_ok=True)

        # Create metadata storage directory
        self.metadata_dir = self.documents_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # Initialize text splitter with RAG settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(f"DocumentProcessor initialized with documents_dir: {self.documents_dir}")

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def extract_text_from_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF file using PyMuPDF (fallback to PyPDF2)"""
        text = ""
        metadata = {"pages": 0, "extraction_method": ""}

        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(file_path)
            metadata["pages"] = len(doc)
            metadata["extraction_method"] = "PyMuPDF"

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"  # Add page separator

            doc.close()

        except Exception as e1:
            logger.warning(f"PyMuPDF extraction failed: {e1}, trying PyPDF2")

            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(pdf_reader.pages)
                    metadata["extraction_method"] = "PyPDF2"

                    for page in pdf_reader.pages:
                        text += page.extract_text()
                        text += "\n\n"

            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: PyMuPDF={e1}, PyPDF2={e2}")
                raise ValueError(f"Could not extract text from PDF: {e2}")

        return text.strip(), metadata

    def extract_text_from_txt(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text from TXT/MD file"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            encoding = "utf-8"
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                encoding = "latin-1"
            except Exception as e:
                logger.error(f"Could not read text file {file_path}: {e}")
                raise ValueError(f"Could not read text file: {e}")

        metadata = {
            "encoding": encoding,
            "line_count": len(text.split('\n'))
        }

        return text, metadata

    def process_file(self, file_path: Path, custom_metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process a single file and return Document object"""

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size
        max_size_bytes = settings.rag.max_file_size_mb * 1024 * 1024

        if file_size > max_size_bytes:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB > {settings.rag.max_file_size_mb}MB")

        # Get file info
        file_type = file_path.suffix.lower().lstrip('.')
        if file_type not in settings.rag.supported_formats:
            raise ValueError(f"Unsupported file format: {file_type}")

        # Calculate file hash for duplicate detection
        file_hash = self.get_file_hash(file_path)

        # Check if we already processed this file
        existing_doc = self.get_document_by_hash(file_hash)
        if existing_doc:
            logger.info(f"File already processed: {file_path.name} (hash: {file_hash[:8]}...)")
            return existing_doc

        # Extract text based on file type
        if file_type == "pdf":
            text, extraction_metadata = self.extract_text_from_pdf(file_path)
        else:  # txt, md
            text, extraction_metadata = self.extract_text_from_txt(file_path)

        if len(text.strip()) < settings.rag.min_chunk_size:
            raise ValueError(f"File content too short: {len(text)} characters")

        # Create document ID
        document_id = f"doc_{file_hash[:16]}_{int(datetime.now().timestamp())}"

        # Combine metadata
        metadata = {
            "extraction_metadata": extraction_metadata,
            "original_filename": file_path.name,
            **(custom_metadata or {})
        }

        # Create document object
        document = Document(
            document_id=document_id,
            filename=file_path.name,
            file_type=file_type,
            content=text,
            file_size=file_size,
            file_hash=file_hash,
            metadata=metadata
        )

        # Create chunks
        chunks = self.create_chunks(document)
        for chunk in chunks:
            document.add_chunk(chunk)

        # Save document metadata
        self.save_document_metadata(document)

        logger.info(f"Processed document: {file_path.name} -> {len(chunks)} chunks")
        return document

    def create_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create text chunks from document content"""

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(document.content)

        chunks = []
        current_pos = 0

        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) < settings.rag.min_chunk_size:
                continue

            # Find the chunk position in original text
            start_pos = document.content.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(chunk_text)
            current_pos = end_pos

            # Create chunk ID
            chunk_id = f"{document.document_id}_chunk_{i:04d}"

            # Estimate page number for PDFs
            page_number = None
            if document.file_type == "pdf" and "pages" in document.metadata.get("extraction_metadata", {}):
                total_chars = len(document.content)
                estimated_page = int((start_pos / total_chars) * document.metadata["extraction_metadata"]["pages"]) + 1
                page_number = min(estimated_page, document.metadata["extraction_metadata"]["pages"])

            chunk = DocumentChunk(
                content=chunk_text.strip(),
                chunk_id=chunk_id,
                document_id=document.document_id,
                page_number=page_number,
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "chunk_index": i,
                    "chunk_length": len(chunk_text)
                }
            )

            chunks.append(chunk)

        return chunks

    def save_document_metadata(self, document: Document):
        """Save document metadata to JSON file"""
        metadata_file = self.metadata_dir / f"{document.document_id}.json"

        # Save document metadata (without full content to save space)
        doc_dict = document.to_dict()
        doc_dict["content"] = f"<{len(document.content)} characters>"  # Summary only

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Check if document with this hash already exists"""
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    if doc_data.get("file_hash") == file_hash:
                        # Load full document (metadata only contains summary)
                        return self.load_document(doc_data["document_id"])
            except Exception as e:
                logger.warning(f"Could not read metadata file {metadata_file}: {e}")

        return None

    def load_document(self, document_id: str) -> Optional[Document]:
        """Load a document by ID (this would typically load from database)"""
        # For now, return None as we don't store full content in metadata
        # In a real implementation, this would load from a database
        return None

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        documents = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                    documents.append(doc_data)
            except Exception as e:
                logger.warning(f"Could not read metadata file {metadata_file}: {e}")

        # Sort by creation time (newest first)
        documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return documents

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its metadata"""
        metadata_file = self.metadata_dir / f"{document_id}.json"

        if metadata_file.exists():
            try:
                metadata_file.unlink()
                logger.info(f"Deleted document metadata: {document_id}")
                return True
            except Exception as e:
                logger.error(f"Could not delete document metadata {document_id}: {e}")
                return False

        return False
