"""
Pydantic models for API request/response schemas with chat support
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class GenerationRequest(BaseModel):
    """Request model for text generation"""
    
    prompt: str = Field(..., description="Input text prompt for generation", min_length=1, max_length=2000)
    max_length: Optional[int] = Field(100, description="Maximum length of generated text", ge=1, le=512)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    num_return_sequences: Optional[int] = Field(1, description="Number of sequences to return", ge=1, le=5)
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty or only whitespace')
        return v.strip()


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    
    generated_text: List[str] = Field(..., description="Generated text sequences")
    prompt: str = Field(..., description="Original input prompt")
    model_name: str = Field(..., description="Name of the model used")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    parameters: dict = Field(..., description="Generation parameters used")


class HealthResponse(BaseModel):
    """Response model for health check"""
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    uptime: float = Field(..., description="Server uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    
    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type/architecture of the model")
    model_size: str = Field(..., description="Approximate model size")
    device: str = Field(..., description="Device the model is running on")
    max_length: int = Field(..., description="Maximum sequence length supported")
    loaded: bool = Field(..., description="Whether the model is loaded")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")


# Chat-specific schemas

class ChatMessage(BaseModel):
    """Individual chat message"""
    
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content", min_length=1, max_length=8000)
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role must be system, user, or assistant')
        return v


class ChatRequest(BaseModel):
    """Request model for chat interaction"""
    
    message: str = Field(..., description="User message", min_length=1, max_length=8000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    model_name: Optional[str] = Field(None, description="Model to use for chat")
    max_tokens: Optional[int] = Field(200, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: Optional[float] = Field(0.6, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    system_prompt: Optional[str] = Field(None, description="System prompt for new conversations")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or only whitespace')
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat interaction"""
    
    response: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation ID")
    model_name: str = Field(..., description="Model used for generation")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    message_count: int = Field(..., description="Total messages in conversation")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters used")


class ConversationResponse(BaseModel):
    """Response model for conversation data"""
    
    id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    message_count: Dict[str, int] = Field(..., description="Count by message role")
    max_length: int = Field(..., description="Maximum conversation length")


class ModelSwitchRequest(BaseModel):
    """Request model for switching models"""
    
    model_name: str = Field(..., description="Target model name")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if not v.strip():
            raise ValueError('Model name cannot be empty')
        return v.strip()


class ModelListResponse(BaseModel):
    """Response model for available models"""
    
    models: List[Dict[str, Any]] = Field(..., description="Available models with details")
    chat_models: List[str] = Field(..., description="Models that support chat")
    current_model: str = Field(..., description="Currently loaded model")


class NewConversationRequest(BaseModel):
    """Request model for creating new conversation"""
    
    system_prompt: Optional[str] = Field(
        "You are a helpful AI assistant.", 
        description="System prompt for the conversation"
    )
    model_name: Optional[str] = Field(None, description="Model to use")
    max_length: Optional[int] = Field(50, description="Max conversation length", ge=1, le=200)


# RAG-specific schemas

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    
    filename: str = Field(..., description="Original filename", min_length=1, max_length=255)
    file_type: str = Field(..., description="File type (pdf, txt, md)")
    content: str = Field(..., description="Document content", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed_types = ['pdf', 'txt', 'md']
        if v.lower() not in allowed_types:
            raise ValueError(f'File type must be one of: {", ".join(allowed_types)}')
        return v.lower()


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    
    document_id: str = Field(..., description="Generated document ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(..., description="Processing status")


class DocumentInfo(BaseModel):
    """Information about a stored document"""
    
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: str = Field(..., description="Creation timestamp")
    sample_content: str = Field(..., description="Sample content preview")


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    
    documents: List[DocumentInfo] = Field(..., description="List of stored documents")
    total_count: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")


class SearchRequest(BaseModel):
    """Request model for vector similarity search"""
    
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=20)
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result"""
    
    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Chunk content")
    similarity_score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Result rank")
    page_number: Optional[int] = Field(None, description="Page number (for PDFs)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SearchResponse(BaseModel):
    """Response model for search results"""
    
    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")


class RAGChatRequest(BaseModel):
    """Request model for RAG-enhanced chat"""
    
    message: str = Field(..., description="User message", min_length=1, max_length=8000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    model_name: Optional[str] = Field(None, description="Model to use for chat")
    max_tokens: Optional[int] = Field(300, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: Optional[float] = Field(0.6, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter", ge=1, le=100)
    system_prompt: Optional[str] = Field(None, description="System prompt for new conversations")
    
    # RAG-specific parameters
    use_rag: Optional[bool] = Field(True, description="Enable RAG for this request")
    max_chunks: Optional[int] = Field(5, description="Maximum chunks to include", ge=1, le=10)
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(None, description="Filter by specific document IDs")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or only whitespace')
        return v.strip()


class RAGChatResponse(BaseModel):
    """Response model for RAG-enhanced chat"""
    
    response: str = Field(..., description="Assistant's response")
    conversation_id: str = Field(..., description="Conversation ID")
    model_name: str = Field(..., description="Model used for generation")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    message_count: int = Field(..., description="Total messages in conversation")
    parameters: Dict[str, Any] = Field(..., description="Generation parameters used")
    
    # RAG-specific fields
    rag_used: bool = Field(..., description="Whether RAG was used")
    search_results: Optional[List[SearchResult]] = Field(None, description="Retrieved chunks")
    context_length: Optional[int] = Field(None, description="Length of RAG context used")


class VectorStoreStatsResponse(BaseModel):
    """Response model for vector store statistics"""
    
    total_chunks: int = Field(..., description="Total number of chunks")
    total_documents: int = Field(..., description="Total number of documents")
    index_size_mb: float = Field(..., description="Index size in megabytes")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")
    embeddings_model: str = Field(..., description="Embeddings model name")
    embedding_dimension: Optional[int] = Field(None, description="Embedding vector dimension")


class DocumentDeleteRequest(BaseModel):
    """Request model for document deletion"""
    
    document_id: str = Field(..., description="Document ID to delete", min_length=1)
    hard_delete: Optional[bool] = Field(False, description="Permanently delete (vs soft delete)")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion"""
    
    document_id: str = Field(..., description="Deleted document ID")
    status: str = Field(..., description="Deletion status")
    deleted_chunks: int = Field(..., description="Number of chunks deleted")
    message: str = Field(..., description="Success message")


class DocumentChunkInfo(BaseModel):
    """Information about a document chunk"""
    
    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk content")
    page_number: Optional[int] = Field(None, description="Page number (for PDFs)")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")
    chunk_index: int = Field(..., description="Chunk index in document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional chunk metadata")


class DocumentChunksResponse(BaseModel):
    """Response model for document chunks preview"""
    
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    chunks: List[DocumentChunkInfo] = Field(..., description="Document chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    chunk_size: int = Field(..., description="Chunk size used")
    chunk_overlap: int = Field(..., description="Chunk overlap used")


class DocumentStoreStatsResponse(BaseModel):
    """Response model for document store statistics"""
    
    total_documents: int = Field(..., description="Total number of documents")
    total_versions: int = Field(..., description="Total number of document versions")
    total_size_bytes: int = Field(..., description="Total storage size in bytes")
    db_size_mb: float = Field(..., description="Database size in megabytes")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class DocumentVersionInfo(BaseModel):
    """Information about a document version"""
    
    version_id: str = Field(..., description="Version ID")
    version: int = Field(..., description="Version number")
    created_at: str = Field(..., description="Version creation timestamp")
    change_summary: Optional[str] = Field(None, description="Summary of changes")
    content_preview: str = Field(..., description="Content preview (first 200 chars)")


class DocumentVersionsResponse(BaseModel):
    """Response model for document version history"""
    
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    current_version: int = Field(..., description="Current version number")
    versions: List[DocumentVersionInfo] = Field(..., description="Version history")
    total_versions: int = Field(..., description="Total number of versions")


class DocumentRestoreRequest(BaseModel):
    """Request model for restoring document version"""
    
    document_id: str = Field(..., description="Document ID", min_length=1)
    version: int = Field(..., description="Version number to restore to", ge=1)
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if not v.strip():
            raise ValueError('Document ID cannot be empty')
        return v.strip()


class DocumentSearchRequest(BaseModel):
    """Request model for advanced document search"""
    
    query: Optional[str] = Field(None, description="Text search query")
    filename_pattern: Optional[str] = Field(None, description="Filename pattern (supports wildcards)")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    date_from: Optional[str] = Field(None, description="Start date (ISO format)")
    date_to: Optional[str] = Field(None, description="End date (ISO format)")
    min_file_size: Optional[int] = Field(None, description="Minimum file size in bytes", ge=0)
    max_file_size: Optional[int] = Field(None, description="Maximum file size in bytes", ge=0)
    metadata_filters: Optional[Dict[str, Any]] = Field(None, description="Metadata key-value filters")
    limit: Optional[int] = Field(20, description="Maximum results to return", ge=1, le=100)
    offset: Optional[int] = Field(0, description="Results to skip", ge=0)
    
    @validator('file_types')
    def validate_file_types(cls, v):
        if v is not None:
            allowed_types = ['pdf', 'txt', 'md']
            for file_type in v:
                if file_type.lower() not in allowed_types:
                    raise ValueError(f'File types must be one of: {", ".join(allowed_types)}')
        return v


class DocumentRecord(BaseModel):
    """Complete document record information"""
    
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., description="File size in bytes")
    file_hash: str = Field(..., description="File content hash")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    version: int = Field(..., description="Current version number")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    is_active: bool = Field(..., description="Whether document is active")


class DocumentSearchResponse(BaseModel):
    """Response model for document search results"""
    
    documents: List[DocumentRecord] = Field(..., description="Matching documents")
    total_count: int = Field(..., description="Total number of matching documents")
    query_time: float = Field(..., description="Search time in seconds")
    filters_applied: Dict[str, Any] = Field(..., description="Applied search filters")