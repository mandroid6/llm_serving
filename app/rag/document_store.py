"""
Document Store module for RAG functionality
Provides persistent document storage and indexing with SQLite backend
"""

import os
import json
import sqlite3
import hashlib
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

from app.core.config import settings
from app.rag.document_processor import Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Database record for a document"""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    file_hash: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "file_hash": self.file_hash,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "is_active": self.is_active
        }


@dataclass
class DocumentVersion:
    """Database record for document version history"""
    version_id: str
    document_id: str
    version: int
    content: str
    metadata: Dict[str, Any]
    change_summary: Optional[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "version_id": self.version_id,
            "document_id": self.document_id,
            "version": self.version,
            "content": self.content,
            "metadata": self.metadata,
            "change_summary": self.change_summary,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class SearchFilter:
    """Filter criteria for document search"""
    document_ids: Optional[List[str]] = None
    filename_pattern: Optional[str] = None
    file_types: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    content_query: Optional[str] = None
    min_file_size: Optional[int] = None
    max_file_size: Optional[int] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class DocumentStore:
    """
    Persistent document storage and indexing system
    Uses SQLite for local development, configurable for other databases
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize document store with database connection"""
        
        # Database configuration
        self.db_path = Path(db_path or settings.rag.document_store_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._local = threading.local()
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_versions": 0,
            "total_size_bytes": 0,
            "last_updated": None,
            "db_size_mb": 0.0
        }
        
        logger.info(f"DocumentStore initialized with database: {self.db_path}")
        
        # Initialize database schema
        self._initialize_database()
        self._update_stats()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode for better concurrency
            self._local.connection = conn
        
        return self._local.connection
    
    @contextmanager
    def _get_cursor(self, transaction: bool = False):
        """Get database cursor with optional transaction"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if transaction:
                cursor.execute("BEGIN TRANSACTION")
            
            yield cursor
            
            if transaction:
                conn.commit()
                
        except Exception as e:
            if transaction:
                conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _initialize_database(self):
        """Create database tables and indexes"""
        with self._lock:
            with self._get_cursor(transaction=True) as cursor:
                
                # Documents table - main document storage
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        file_hash TEXT NOT NULL UNIQUE,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,  -- JSON
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        is_active BOOLEAN NOT NULL DEFAULT 1
                    )
                """)
                
                # Document versions table - version history
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_versions (
                        version_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT NOT NULL,  -- JSON
                        change_summary TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
                    )
                """)
                
                # Document metadata table - flexible key-value pairs for rich querying
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        document_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        value_type TEXT NOT NULL,  -- 'string', 'number', 'boolean', 'date'
                        PRIMARY KEY (document_id, key),
                        FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents (filename)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents (file_type)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents (file_hash)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents (created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents (updated_at)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_is_active ON documents (is_active)",
                    "CREATE INDEX IF NOT EXISTS idx_document_versions_document_id ON document_versions (document_id)",
                    "CREATE INDEX IF NOT EXISTS idx_document_versions_version ON document_versions (version)",
                    "CREATE INDEX IF NOT EXISTS idx_document_metadata_key ON document_metadata (key)",
                    "CREATE INDEX IF NOT EXISTS idx_document_metadata_value ON document_metadata (value)",
                    "CREATE INDEX IF NOT EXISTS idx_document_metadata_key_value ON document_metadata (key, value)"
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                logger.info("Database schema initialized successfully")
    
    def add_document(self, document: Document, replace_existing: bool = False) -> bool:
        """
        Add a document to the store
        
        Args:
            document: Document object to store
            replace_existing: If True, replace existing document with same hash
            
        Returns:
            bool: True if successful, False otherwise
        """
        with self._lock:
            try:
                with self._get_cursor(transaction=True) as cursor:
                    
                    # Check if document with this hash already exists
                    cursor.execute(
                        "SELECT document_id, version FROM documents WHERE file_hash = ? AND is_active = 1",
                        (document.file_hash,)
                    )
                    existing = cursor.fetchone()
                    
                    if existing and not replace_existing:
                        logger.warning(f"Document with hash {document.file_hash} already exists")
                        return False
                    
                    current_time = datetime.now(timezone.utc)
                    
                    if existing and replace_existing:
                        # Update existing document (create new version)
                        existing_doc_id = existing['document_id']
                        new_version = existing['version'] + 1
                        
                        # Save current version to history
                        cursor.execute(
                            "SELECT * FROM documents WHERE document_id = ?",
                            (existing_doc_id,)
                        )
                        current_doc = cursor.fetchone()
                        
                        version_id = f"{existing_doc_id}_v{current_doc['version']}"
                        cursor.execute("""
                            INSERT INTO document_versions 
                            (version_id, document_id, version, content, metadata, change_summary, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            version_id,
                            existing_doc_id,
                            current_doc['version'],
                            current_doc['content'],
                            current_doc['metadata'],
                            "Document updated",
                            current_doc['updated_at']
                        ))
                        
                        # Update document with new content
                        cursor.execute("""
                            UPDATE documents SET
                                filename = ?, content = ?, metadata = ?, 
                                updated_at = ?, version = ?
                            WHERE document_id = ?
                        """, (
                            document.filename,
                            document.content,
                            json.dumps(document.metadata),
                            current_time.isoformat(),
                            new_version,
                            existing_doc_id
                        ))
                        
                        # Update metadata table
                        self._update_metadata_table(cursor, existing_doc_id, document.metadata)
                        
                        logger.info(f"Updated document {existing_doc_id} to version {new_version}")
                        document.document_id = existing_doc_id
                        
                    else:
                        # Insert new document
                        cursor.execute("""
                            INSERT INTO documents 
                            (document_id, filename, file_type, file_size, file_hash, 
                             content, metadata, created_at, updated_at, version, is_active)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            document.document_id,
                            document.filename,
                            document.file_type,
                            document.file_size,
                            document.file_hash,
                            document.content,
                            json.dumps(document.metadata),
                            current_time.isoformat(),
                            current_time.isoformat(),
                            1,
                            True
                        ))
                        
                        # Add to metadata table
                        self._update_metadata_table(cursor, document.document_id, document.metadata)
                        
                        logger.info(f"Added new document {document.document_id}")
                    
                    self._update_stats()
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to add document {document.document_id}: {e}")
                return False
    
    def _update_metadata_table(self, cursor: sqlite3.Cursor, document_id: str, metadata: Dict[str, Any]):
        """Update the metadata table for a document"""
        
        # Clear existing metadata
        cursor.execute("DELETE FROM document_metadata WHERE document_id = ?", (document_id,))
        
        # Insert new metadata
        for key, value in metadata.items():
            if isinstance(value, dict) or isinstance(value, list):
                value_str = json.dumps(value)
                value_type = "json"
            elif isinstance(value, bool):
                value_str = str(value).lower()
                value_type = "boolean"
            elif isinstance(value, (int, float)):
                value_str = str(value)
                value_type = "number"
            elif isinstance(value, datetime):
                value_str = value.isoformat()
                value_type = "date"
            else:
                value_str = str(value)
                value_type = "string"
            
            cursor.execute("""
                INSERT INTO document_metadata (document_id, key, value, value_type)
                VALUES (?, ?, ?, ?)
            """, (document_id, key, value_str, value_type))
    
    def get_document(self, document_id: str, include_content: bool = True) -> Optional[DocumentRecord]:
        """
        Get a document by ID
        
        Args:
            document_id: Document ID to retrieve
            include_content: Whether to include full content (for large documents)
            
        Returns:
            DocumentRecord or None if not found
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    
                    if include_content:
                        cursor.execute("""
                            SELECT * FROM documents 
                            WHERE document_id = ? AND is_active = 1
                        """, (document_id,))
                    else:
                        cursor.execute("""
                            SELECT document_id, filename, file_type, file_size, file_hash,
                                   metadata, created_at, updated_at, version, is_active
                            FROM documents 
                            WHERE document_id = ? AND is_active = 1
                        """, (document_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    return DocumentRecord(
                        document_id=row['document_id'],
                        filename=row['filename'],
                        file_type=row['file_type'],
                        file_size=row['file_size'],
                        file_hash=row['file_hash'],
                        content=row.get('content', ''),
                        metadata=json.loads(row['metadata']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        version=row['version'],
                        is_active=bool(row['is_active'])
                    )
                    
            except Exception as e:
                logger.error(f"Failed to get document {document_id}: {e}")
                return None
    
    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        """Get a document by file hash"""
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM documents 
                        WHERE file_hash = ? AND is_active = 1
                    """, (file_hash,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    return DocumentRecord(
                        document_id=row['document_id'],
                        filename=row['filename'],
                        file_type=row['file_type'],
                        file_size=row['file_size'],
                        file_hash=row['file_hash'],
                        content=row['content'],
                        metadata=json.loads(row['metadata']),
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        version=row['version'],
                        is_active=bool(row['is_active'])
                    )
                    
            except Exception as e:
                logger.error(f"Failed to get document by hash {file_hash}: {e}")
                return None