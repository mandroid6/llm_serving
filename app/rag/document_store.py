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
                        content=row['content'] if 'content' in row.keys() else '',
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
    
    def search_documents(self, search_filter: SearchFilter) -> List[DocumentRecord]:
        """
        Search documents using filter criteria
        
        Args:
            search_filter: SearchFilter with criteria
            
        Returns:
            List of matching DocumentRecord objects
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    
                    # Build query and parameters
                    conditions = []
                    params = []
                    
                    # Always filter active documents
                    conditions.append("is_active = 1")
                    
                    # Document IDs filter
                    if search_filter.document_ids:
                        placeholders = ','.join(['?'] * len(search_filter.document_ids))
                        conditions.append(f"document_id IN ({placeholders})")
                        params.extend(search_filter.document_ids)
                    
                    # Filename pattern filter
                    if search_filter.filename_pattern:
                        conditions.append("filename LIKE ?")
                        params.append(f"%{search_filter.filename_pattern}%")
                    
                    # File types filter
                    if search_filter.file_types:
                        placeholders = ','.join(['?'] * len(search_filter.file_types))
                        conditions.append(f"file_type IN ({placeholders})")
                        params.extend(search_filter.file_types)
                    
                    # Date range filters
                    if search_filter.date_from:
                        conditions.append("created_at >= ?")
                        params.append(search_filter.date_from.isoformat())
                    
                    if search_filter.date_to:
                        conditions.append("created_at <= ?")
                        params.append(search_filter.date_to.isoformat())
                    
                    # File size filters
                    if search_filter.min_file_size is not None:
                        conditions.append("file_size >= ?")
                        params.append(search_filter.min_file_size)
                    
                    if search_filter.max_file_size is not None:
                        conditions.append("file_size <= ?")
                        params.append(search_filter.max_file_size)
                    
                    # Content search (FTS if available, otherwise LIKE)
                    if search_filter.content_query:
                        conditions.append("content LIKE ?")
                        params.append(f"%{search_filter.content_query}%")
                    
                    # Metadata filters
                    if search_filter.metadata_filters:
                        for key, value in search_filter.metadata_filters.items():
                            conditions.append("""
                                document_id IN (
                                    SELECT document_id FROM document_metadata 
                                    WHERE key = ? AND value = ?
                                )
                            """)
                            params.extend([key, str(value)])
                    
                    # Build final query
                    where_clause = " AND ".join(conditions) if conditions else "1=1"
                    
                    query = f"""
                        SELECT * FROM documents 
                        WHERE {where_clause}
                        ORDER BY updated_at DESC
                    """
                    
                    # Add limit and offset
                    if search_filter.limit:
                        query += " LIMIT ?"
                        params.append(search_filter.limit)
                        
                        if search_filter.offset:
                            query += " OFFSET ?"
                            params.append(search_filter.offset)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert to DocumentRecord objects
                    documents = []
                    for row in rows:
                        documents.append(DocumentRecord(
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
                        ))
                    
                    logger.info(f"Found {len(documents)} documents matching search criteria")
                    return documents
                    
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []
    
    def update_document(self, document: Document, change_summary: Optional[str] = None) -> bool:
        """
        Update an existing document (creates new version)
        
        Args:
            document: Updated Document object
            change_summary: Optional description of changes
            
        Returns:
            bool: True if successful
        """
        return self.add_document(document, replace_existing=True)
    
    def delete_document(self, document_id: str, soft_delete: bool = True) -> bool:
        """
        Delete a document from the store
        
        Args:
            document_id: Document ID to delete
            soft_delete: If True, mark as inactive; if False, physically delete
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            try:
                with self._get_cursor(transaction=True) as cursor:
                    
                    # Check if document exists
                    cursor.execute(
                        "SELECT document_id FROM documents WHERE document_id = ? AND is_active = 1",
                        (document_id,)
                    )
                    
                    if not cursor.fetchone():
                        logger.warning(f"Document {document_id} not found or already deleted")
                        return False
                    
                    if soft_delete:
                        # Soft delete - mark as inactive
                        cursor.execute(
                            "UPDATE documents SET is_active = 0, updated_at = ? WHERE document_id = ?",
                            (datetime.now(timezone.utc).isoformat(), document_id)
                        )
                        logger.info(f"Soft deleted document {document_id}")
                    else:
                        # Hard delete - remove from database
                        cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
                        # Metadata and versions will be cascade deleted due to foreign keys
                        logger.info(f"Hard deleted document {document_id}")
                    
                    self._update_stats()
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to delete document {document_id}: {e}")
                return False
    
    def list_documents(
        self, 
        include_content: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: str = "updated_at DESC"
    ) -> List[DocumentRecord]:
        """
        List all active documents
        
        Args:
            include_content: Whether to include full content
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            order_by: Sort order (e.g., "created_at DESC", "filename ASC")
            
        Returns:
            List of DocumentRecord objects
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    
                    # Select columns based on include_content
                    if include_content:
                        select_clause = "*"
                    else:
                        select_clause = """
                            document_id, filename, file_type, file_size, file_hash,
                            metadata, created_at, updated_at, version, is_active
                        """
                    
                    query = f"""
                        SELECT {select_clause} FROM documents 
                        WHERE is_active = 1 
                        ORDER BY {order_by}
                    """
                    
                    params = []
                    if limit:
                        query += " LIMIT ?"
                        params.append(limit)
                        
                        if offset:
                            query += " OFFSET ?"
                            params.append(offset)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    documents = []
                    for row in rows:
                        documents.append(DocumentRecord(
                            document_id=row['document_id'],
                            filename=row['filename'],
                            file_type=row['file_type'],
                            file_size=row['file_size'],
                            file_hash=row['file_hash'],
                            content=row['content'] if 'content' in row.keys() else '',
                            metadata=json.loads(row['metadata']),
                            created_at=datetime.fromisoformat(row['created_at']),
                            updated_at=datetime.fromisoformat(row['updated_at']),
                            version=row['version'],
                            is_active=bool(row['is_active'])
                        ))
                    
                    return documents
                    
            except Exception as e:
                logger.error(f"Failed to list documents: {e}")
                return []
    
    def get_document_versions(self, document_id: str) -> List[DocumentVersion]:
        """
        Get all versions of a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of DocumentVersion objects sorted by version (newest first)
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM document_versions 
                        WHERE document_id = ? 
                        ORDER BY version DESC
                    """, (document_id,))
                    
                    rows = cursor.fetchall()
                    versions = []
                    
                    for row in rows:
                        versions.append(DocumentVersion(
                            version_id=row['version_id'],
                            document_id=row['document_id'],
                            version=row['version'],
                            content=row['content'],
                            metadata=json.loads(row['metadata']),
                            change_summary=row['change_summary'],
                            created_at=datetime.fromisoformat(row['created_at'])
                        ))
                    
                    return versions
                    
            except Exception as e:
                logger.error(f"Failed to get document versions for {document_id}: {e}")
                return []
    
    def get_document_version(self, document_id: str, version: int) -> Optional[DocumentVersion]:
        """
        Get a specific version of a document
        
        Args:
            document_id: Document ID
            version: Version number
            
        Returns:
            DocumentVersion object or None
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM document_versions 
                        WHERE document_id = ? AND version = ?
                    """, (document_id, version))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    return DocumentVersion(
                        version_id=row['version_id'],
                        document_id=row['document_id'],
                        version=row['version'],
                        content=row['content'],
                        metadata=json.loads(row['metadata']),
                        change_summary=row['change_summary'],
                        created_at=datetime.fromisoformat(row['created_at'])
                    )
                    
            except Exception as e:
                logger.error(f"Failed to get document version {document_id} v{version}: {e}")
                return None
    
    def restore_document_version(self, document_id: str, version: int) -> bool:
        """
        Restore a document to a specific version
        
        Args:
            document_id: Document ID
            version: Version number to restore to
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            try:
                # Get the version to restore
                version_doc = self.get_document_version(document_id, version)
                if not version_doc:
                    logger.error(f"Version {version} not found for document {document_id}")
                    return False
                
                with self._get_cursor(transaction=True) as cursor:
                    # Get current document info
                    cursor.execute(
                        "SELECT filename, file_type, file_size, file_hash FROM documents WHERE document_id = ?",
                        (document_id,)
                    )
                    current_doc = cursor.fetchone()
                    if not current_doc:
                        logger.error(f"Document {document_id} not found")
                        return False
                    
                    # Save current version to history first
                    cursor.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,))
                    current_full = cursor.fetchone()
                    
                    version_id = f"{document_id}_v{current_full['version']}"
                    cursor.execute("""
                        INSERT INTO document_versions 
                        (version_id, document_id, version, content, metadata, change_summary, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version_id,
                        document_id,
                        current_full['version'],
                        current_full['content'],
                        current_full['metadata'],
                        f"Backup before restore to v{version}",
                        current_full['updated_at']
                    ))
                    
                    # Update document with restored content
                    new_version = current_full['version'] + 1
                    current_time = datetime.now(timezone.utc)
                    
                    cursor.execute("""
                        UPDATE documents SET
                            content = ?, metadata = ?, updated_at = ?, version = ?
                        WHERE document_id = ?
                    """, (
                        version_doc.content,
                        json.dumps(version_doc.metadata),
                        current_time.isoformat(),
                        new_version,
                        document_id
                    ))
                    
                    # Update metadata table
                    self._update_metadata_table(cursor, document_id, version_doc.metadata)
                    
                    logger.info(f"Restored document {document_id} to version {version} (new version {new_version})")
                    self._update_stats()
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to restore document {document_id} to version {version}: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics"""
        return self.stats.copy()
    
    def _update_stats(self):
        """Update internal statistics"""
        with self._get_cursor() as cursor:
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM documents WHERE is_active = 1")
            self.stats["total_documents"] = cursor.fetchone()[0]
            
            # Count versions
            cursor.execute("SELECT COUNT(*) FROM document_versions")
            self.stats["total_versions"] = cursor.fetchone()[0]
            
            # Calculate total size
            cursor.execute("SELECT SUM(file_size) FROM documents WHERE is_active = 1")
            result = cursor.fetchone()[0]
            self.stats["total_size_bytes"] = result or 0
            
            # Update timestamp
            self.stats["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Calculate database size
            try:
                db_size_bytes = self.db_path.stat().st_size
                self.stats["db_size_mb"] = db_size_bytes / (1024 * 1024)
            except Exception:
                self.stats["db_size_mb"] = 0.0
    
    def count_documents(self, file_types: Optional[List[str]] = None) -> int:
        """
        Count active documents, optionally filtered by file types
        
        Args:
            file_types: Optional list of file types to filter by
            
        Returns:
            int: Number of documents
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    if file_types:
                        placeholders = ','.join(['?'] * len(file_types))
                        cursor.execute(
                            f"SELECT COUNT(*) FROM documents WHERE is_active = 1 AND file_type IN ({placeholders})",
                            file_types
                        )
                    else:
                        cursor.execute("SELECT COUNT(*) FROM documents WHERE is_active = 1")
                    
                    return cursor.fetchone()[0]
                    
            except Exception as e:
                logger.error(f"Failed to count documents: {e}")
                return 0
    
    # Integration methods for DocumentProcessor and VectorStore coordination
    
    def sync_with_document_processor(self, processor: 'DocumentProcessor') -> Dict[str, Any]:
        """
        Synchronize with DocumentProcessor to ensure consistency
        
        Args:
            processor: DocumentProcessor instance
            
        Returns:
            Dictionary with sync results
        """
        results = {
            "documents_added": 0,
            "documents_updated": 0,
            "errors": []
        }
        
        try:
            # Get documents from processor metadata
            processor_docs = processor.list_documents()
            
            for doc_data in processor_docs:
                try:
                    # Check if document exists in our store
                    existing = self.get_document_by_hash(doc_data["file_hash"])
                    
                    if not existing:
                        # Document is in processor but not in our store
                        # This would typically require the full Document object
                        # For now, just log it
                        logger.warning(f"Document {doc_data['document_id']} exists in processor but not in document store")
                    
                except Exception as e:
                    error_msg = f"Error syncing document {doc_data.get('document_id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            logger.info(f"Sync with DocumentProcessor completed: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Failed to sync with DocumentProcessor: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
    
    def sync_with_vector_store(self, vector_store: 'VectorStore') -> Dict[str, Any]:
        """
        Synchronize with VectorStore to ensure consistency
        
        Args:
            vector_store: VectorStore instance
            
        Returns:
            Dictionary with sync results
        """
        results = {
            "documents_in_sync": 0,
            "documents_missing_from_vector_store": [],
            "documents_missing_from_document_store": [],
            "errors": []
        }
        
        try:
            # Get all active documents from document store
            store_docs = {doc.document_id: doc for doc in self.list_documents(include_content=False)}
            
            # Get documents from vector store
            vector_docs = vector_store.list_documents()
            vector_doc_ids = {doc["document_id"] for doc in vector_docs}
            
            # Find documents in document store but not in vector store
            for doc_id in store_docs:
                if doc_id not in vector_doc_ids:
                    results["documents_missing_from_vector_store"].append(doc_id)
            
            # Find documents in vector store but not in document store
            for doc_id in vector_doc_ids:
                if doc_id not in store_docs:
                    results["documents_missing_from_document_store"].append(doc_id)
            
            # Count documents that are in sync
            results["documents_in_sync"] = len(vector_doc_ids.intersection(store_docs.keys()))
            
            logger.info(f"Sync with VectorStore completed: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Failed to sync with VectorStore: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
    
    # Backup and restore methods
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the document store database
        
        Args:
            backup_path: Path for backup file (auto-generated if None)
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            try:
                if backup_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"{self.db_path.parent}/document_store_backup_{timestamp}.db"
                
                backup_path = Path(backup_path)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create backup using SQLite backup API
                source_conn = self._get_connection()
                
                # Use sqlite3.connect to create backup connection
                backup_conn = sqlite3.connect(str(backup_path))
                
                # Perform backup
                source_conn.backup(backup_conn)
                backup_conn.close()
                
                logger.info(f"Database backed up to: {backup_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                return False
    
    def restore_database(self, backup_path: str, confirm: bool = False) -> bool:
        """
        Restore document store database from backup
        
        Args:
            backup_path: Path to backup file
            confirm: Must be True to proceed (safety check)
            
        Returns:
            bool: True if successful
        """
        if not confirm:
            logger.error("restore_database requires confirm=True as safety check")
            return False
            
        with self._lock:
            try:
                backup_path = Path(backup_path)
                if not backup_path.exists():
                    logger.error(f"Backup file not found: {backup_path}")
                    return False
                
                # Close existing connections
                if hasattr(self._local, 'connection'):
                    self._local.connection.close()
                    delattr(self._local, 'connection')
                
                # Create backup of current database
                current_backup = f"{self.db_path}.before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import shutil
                shutil.copy2(self.db_path, current_backup)
                logger.info(f"Current database backed up to: {current_backup}")
                
                # Replace current database with backup
                shutil.copy2(backup_path, self.db_path)
                
                # Reinitialize
                self._initialize_database()
                self._update_stats()
                
                logger.info(f"Database restored from: {backup_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to restore from backup: {e}")
                return False
    
    def export_documents(self, export_path: str, format: str = "json") -> bool:
        """
        Export all documents to a file
        
        Args:
            export_path: Path for export file
            format: Export format ("json" or "csv")
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            try:
                export_path = Path(export_path)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                
                documents = self.list_documents(include_content=True)
                
                if format.lower() == "json":
                    export_data = {
                        "export_timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_documents": len(documents),
                        "documents": [doc.to_dict() for doc in documents]
                    }
                    
                    with open(export_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                elif format.lower() == "csv":
                    import csv
                    
                    with open(export_path, 'w', newline='', encoding='utf-8') as f:
                        if documents:
                            fieldnames = [
                                'document_id', 'filename', 'file_type', 'file_size', 
                                'file_hash', 'created_at', 'updated_at', 'version'
                            ]
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            
                            for doc in documents:
                                row = {
                                    'document_id': doc.document_id,
                                    'filename': doc.filename,
                                    'file_type': doc.file_type,
                                    'file_size': doc.file_size,
                                    'file_hash': doc.file_hash,
                                    'created_at': doc.created_at.isoformat(),
                                    'updated_at': doc.updated_at.isoformat(),
                                    'version': doc.version
                                }
                                writer.writerow(row)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                logger.info(f"Exported {len(documents)} documents to: {export_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to export documents: {e}")
                return False
    
    def cleanup_old_versions(self, keep_versions: int = 5) -> int:
        """
        Clean up old document versions, keeping only the most recent N versions
        
        Args:
            keep_versions: Number of versions to keep per document
            
        Returns:
            int: Number of versions deleted
        """
        with self._lock:
            try:
                deleted_count = 0
                
                with self._get_cursor(transaction=True) as cursor:
                    # Get all documents
                    cursor.execute("SELECT document_id FROM documents WHERE is_active = 1")
                    document_ids = [row[0] for row in cursor.fetchall()]
                    
                    for doc_id in document_ids:
                        # Get versions for this document, ordered by version desc
                        cursor.execute("""
                            SELECT version_id FROM document_versions 
                            WHERE document_id = ? 
                            ORDER BY version DESC
                            LIMIT -1 OFFSET ?
                        """, (doc_id, keep_versions))
                        
                        old_versions = [row[0] for row in cursor.fetchall()]
                        
                        # Delete old versions
                        for version_id in old_versions:
                            cursor.execute("DELETE FROM document_versions WHERE version_id = ?", (version_id,))
                            deleted_count += 1
                
                logger.info(f"Cleaned up {deleted_count} old document versions")
                return deleted_count
                
            except Exception as e:
                logger.error(f"Failed to cleanup old versions: {e}")
                return 0
    
    def vacuum_database(self) -> bool:
        """
        Vacuum the database to reclaim space and optimize performance
        
        Returns:
            bool: True if successful
        """
        with self._lock:
            try:
                with self._get_cursor() as cursor:
                    cursor.execute("VACUUM")
                
                self._update_stats()
                logger.info("Database vacuumed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to vacuum database: {e}")
                return False
    
    def close(self):
        """Close database connections and cleanup"""
        try:
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
                delattr(self._local, 'connection')
            logger.info("DocumentStore closed")
        except Exception as e:
            logger.error(f"Error closing DocumentStore: {e}")


# Global document store instance
document_store = DocumentStore()


def get_document_store() -> DocumentStore:
    """Get the global document store instance"""
    return document_store


# Convenience functions
def add_document(document: Document, replace_existing: bool = False) -> bool:
    """Convenience function to add document using global store"""
    return document_store.add_document(document, replace_existing)


def get_document(document_id: str, include_content: bool = True) -> Optional[DocumentRecord]:
    """Convenience function to get document using global store"""
    return document_store.get_document(document_id, include_content)


def search_documents(search_filter: SearchFilter) -> List[DocumentRecord]:
    """Convenience function to search documents using global store"""
    return document_store.search_documents(search_filter)


def delete_document(document_id: str, soft_delete: bool = True) -> bool:
    """Convenience function to delete document using global store"""
    return document_store.delete_document(document_id, soft_delete)


def list_documents(include_content: bool = False, limit: Optional[int] = None) -> List[DocumentRecord]:
    """Convenience function to list documents using global store"""
    return document_store.list_documents(include_content, limit)