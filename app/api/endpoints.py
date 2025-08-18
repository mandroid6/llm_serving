"""
API endpoints for the LLM serving application with chat support
"""
import time
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from app.models.schemas import (
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ModelSwitchRequest,
    ModelListResponse,
    NewConversationRequest,
    ChatMessage,
    # RAG-specific schemas
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentChunkInfo,
    DocumentChunksResponse,
    DocumentStoreStatsResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    RAGChatRequest,
    RAGChatResponse,
    VectorStoreStatsResponse
)
from app.models.chat_manager import ChatModelManager
from app.models.conversation import Conversation
from app.core.config import settings, get_model_profile, get_chat_models, get_all_models

# RAG imports
from app.rag.document_processor import DocumentProcessor, Document
from app.rag.vector_store import get_vector_store, VectorStore
from app.rag.embeddings import get_embeddings_manager
from app.rag.document_store import get_document_store, DocumentStore, SearchFilter

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Global chat model manager instance
chat_model_manager = ChatModelManager()

# RAG components
document_processor = DocumentProcessor()
vector_store = get_vector_store()
embeddings_manager = get_embeddings_manager()
document_store = get_document_store()

# Active conversations storage (in production, use Redis or database)
active_conversations: Dict[str, Conversation] = {}

# Global variable to track server start time
server_start_time = time.time()


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest) -> GenerationResponse:
    """
    Generate text based on the input prompt
    """
    try:
        logger.info(f"Received generation request for prompt: '{request.prompt[:50]}...'")
        
        # Ensure model is loaded
        if not chat_model_manager.is_loaded:
            logger.info("Model not loaded, loading now...")
            success = await chat_model_manager.load_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to load model"
                )
        
        # Generate text
        result = await chat_model_manager.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            num_return_sequences=request.num_return_sequences,
            do_sample=request.do_sample
        )
        
        logger.info(f"Generation completed in {result['generation_time']:.2f}s")
        
        return GenerationResponse(**result)
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    """
    try:
        current_time = datetime.now().isoformat()
        uptime = time.time() - server_start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=current_time,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about the loaded model
    """
    try:
        model_info = chat_model_manager.get_model_info()
        return ModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/load-model")
async def load_model() -> Dict[str, Any]:
    """
    Explicitly load the model (useful for preloading)
    """
    try:
        if chat_model_manager.is_loaded:
            return {
                "message": "Model already loaded",
                "model_name": chat_model_manager.profile.name,
                "model_id": chat_model_manager.profile.model_id,
                "load_time": chat_model_manager.load_time
            }
        
        logger.info("Loading model...")
        success = await chat_model_manager.load_model()
        
        if success:
            return {
                "message": "Model loaded successfully",
                "model_name": chat_model_manager.profile.name,
                "model_id": chat_model_manager.profile.model_id,
                "load_time": chat_model_manager.load_time
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Failed to load model"
            )
            
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {str(e)}"
        )


# Chat endpoints

@router.post("/chat", response_model=ChatResponse)
async def chat_message(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the chat interface
    """
    try:
        logger.info(f"Received chat request: '{request.message[:50]}...'")
        
        # Switch model if requested
        if request.model_name and request.model_name != chat_model_manager.model_name:
            logger.info(f"Switching to model: {request.model_name}")
            chat_model_manager.switch_model(request.model_name)
        
        # Ensure model is loaded
        if not chat_model_manager.is_loaded:
            logger.info("Model not loaded, loading now...")
            success = await chat_model_manager.load_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to load model"
                )
        
        # Get or create conversation
        conversation = None
        if request.conversation_id and request.conversation_id in active_conversations:
            conversation = active_conversations[request.conversation_id]
        else:
            # Create new conversation
            system_prompt = request.system_prompt or settings.default_system_prompt
            conversation = Conversation(
                system_prompt=system_prompt,
                max_length=settings.max_conversation_length
            )
            active_conversations[conversation.id] = conversation
        
        # Add user message
        conversation.add_user_message(request.message)
        
        # Generate response
        result = await chat_model_manager.chat(
            conversation=conversation,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=request.stream
        )
        
        # Update conversation in storage
        active_conversations[conversation.id] = result["conversation"]
        
        response = ChatResponse(
            response=result["response"],
            conversation_id=conversation.id,
            model_name=result["model_name"],
            generation_time=result["generation_time"],
            message_count=len(conversation.messages),
            parameters=result["parameters"]
        )
        
        logger.info(f"Chat response generated in {result['generation_time']:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.post("/chat/new", response_model=ConversationResponse)
async def new_conversation(request: NewConversationRequest) -> ConversationResponse:
    """
    Create a new conversation
    """
    try:
        # Switch model if requested
        if request.model_name and request.model_name != chat_model_manager.model_name:
            logger.info(f"Switching to model: {request.model_name}")
            chat_model_manager.switch_model(request.model_name)
        
        # Create new conversation
        conversation = Conversation(
            system_prompt=request.system_prompt,
            max_length=request.max_length or settings.max_conversation_length
        )
        
        # Store conversation
        active_conversations[conversation.id] = conversation
        
        # Convert to response format
        messages = [
            ChatMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
            for msg in conversation.messages
        ]
        
        response = ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages,
            message_count=conversation.get_message_count(),
            max_length=conversation.max_length
        )
        
        logger.info(f"New conversation created: {conversation.id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/chat/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    List available models
    """
    try:
        all_models = []
        for model_name in get_all_models():
            profile = get_model_profile(model_name)
            all_models.append({
                "name": model_name,
                "display_name": profile.name,
                "model_id": profile.model_id,
                "supports_chat": profile.supports_chat,
                "description": profile.description,
                "memory_requirement": f"{profile.memory_gb}GB",
                "max_length": profile.max_length
            })
        
        chat_models = get_chat_models()
        
        response = ModelListResponse(
            models=all_models,
            chat_models=chat_models,
            current_model=chat_model_manager.model_name
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/chat/switch-model")
async def switch_model(request: ModelSwitchRequest) -> Dict[str, Any]:
    """
    Switch to a different model
    """
    try:
        if request.model_name not in get_all_models():
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model_name}"
            )
        
        if request.model_name == chat_model_manager.model_name:
            return {
                "message": f"Already using model: {request.model_name}",
                "model_name": chat_model_manager.profile.name,
                "model_id": chat_model_manager.profile.model_id
            }
        
        logger.info(f"Switching from {chat_model_manager.model_name} to {request.model_name}")
        
        # Switch model
        success = chat_model_manager.switch_model(request.model_name)
        if success:
            # Load the new model
            load_success = await chat_model_manager.load_model()
            if load_success:
                return {
                    "message": f"Successfully switched to {request.model_name}",
                    "model_name": chat_model_manager.profile.name,
                    "model_id": chat_model_manager.profile.model_id,
                    "load_time": chat_model_manager.load_time
                }
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load model: {request.model_name}"
                )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch model: {request.model_name}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model switch failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model switch failed: {str(e)}"
        )


@router.get("/chat/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str) -> ConversationResponse:
    """
    Get conversation by ID
    """
    try:
        if conversation_id not in active_conversations:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}"
            )
        
        conversation = active_conversations[conversation_id]
        
        # Convert to response format
        messages = [
            ChatMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
            for msg in conversation.messages
        ]
        
        response = ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=messages,
            message_count=conversation.get_message_count(),
            max_length=conversation.max_length
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation: {str(e)}"
        )


# RAG endpoints

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """
    Upload and process a document for RAG (enhanced with DocumentStore)
    """
    try:
        start_time = time.time()
        logger.info(f"Processing document upload: {request.filename}")
        
        # Create a temporary file-like object for processing
        import tempfile
        import os
        from pathlib import Path
        
        # Determine file extension
        file_ext = request.file_type
        if not request.filename.endswith(f'.{file_ext}'):
            filename = f"{request.filename}.{file_ext}"
        else:
            filename = request.filename
            
        # Create temporary file with content
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_ext}', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(request.content)
            tmp_file.flush()
            
            try:
                # Process the document using DocumentProcessor
                document = document_processor.process_file(
                    Path(tmp_file.name),
                    custom_metadata=request.metadata
                )
                
                # Override filename to use the provided one
                document.filename = filename
                
                # Store in DocumentStore first (persistent storage)
                doc_store_success = document_store.add_document(document)
                if not doc_store_success:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to store document in document store"
                    )
                
                # Add to vector store (for search capabilities)
                vector_success = vector_store.add_document(document)
                if not vector_success:
                    # If vector store fails, we should remove from document store to maintain consistency
                    document_store.delete_document(document.document_id, soft_delete=False)
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to add document to vector store"
                    )
                
                # Save vector store index
                vector_store.save_index()
                
                processing_time = time.time() - start_time
                
                response = DocumentUploadResponse(
                    document_id=document.document_id,
                    filename=document.filename,
                    file_type=document.file_type,
                    file_size=len(request.content),
                    chunk_count=len(document.chunks),
                    processing_time=processing_time,
                    status="success"
                )
                
                logger.info(f"Document processed successfully: {document.document_id} ({len(document.chunks)} chunks)")
                return response
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """
    List all stored documents (enhanced with DocumentStore)
    """
    try:
        # Get documents from DocumentStore (more complete information)
        documents_list = document_store.list_documents(include_content=False)
        
        # Get vector store stats for chunk counts
        vector_stats = vector_store.get_stats()
        
        # Convert to response format
        documents = []
        for doc in documents_list:
            # Get chunks from vector store for this document
            vector_chunks = vector_store.get_chunks_by_document_id(doc.document_id)
            
            documents.append(DocumentInfo(
                document_id=doc.document_id,
                filename=doc.filename,
                file_type=doc.file_type,
                chunk_count=len(vector_chunks),
                created_at=doc.created_at.isoformat(),
                sample_content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            ))
        
        # Get total stats from DocumentStore
        doc_stats = document_store.get_stats()
        
        response = DocumentListResponse(
            documents=documents,
            total_count=doc_stats["total_documents"],
            total_chunks=vector_stats["total_chunks"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str, hard_delete: bool = False) -> DocumentDeleteResponse:
    """
    Delete a document and all its chunks (enhanced with DocumentStore)
    """
    try:
        # Get document info before deletion for response
        doc = document_store.get_document(document_id, include_content=False)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # Get chunk count before deletion
        vector_chunks = vector_store.get_chunks_by_document_id(document_id)
        chunk_count = len(vector_chunks)
        
        # Delete from both stores
        doc_store_success = document_store.delete_document(document_id, soft_delete=not hard_delete)
        vector_success = vector_store.delete_document(document_id)
        
        if not doc_store_success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete document from document store: {document_id}"
            )
        
        if not vector_success:
            logger.warning(f"Failed to delete document from vector store: {document_id}")
            # Don't fail completely if vector store deletion fails, but log it
        
        # Save vector store after deletion
        vector_store.save_index()
        
        delete_type = "hard" if hard_delete else "soft"
        logger.info(f"Document {delete_type} deleted successfully: {document_id}")
        
        return DocumentDeleteResponse(
            document_id=document_id,
            status="success",
            deleted_chunks=chunk_count,
            message=f"Document {document_id} {delete_type} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.get("/documents/{document_id}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str) -> DocumentChunksResponse:
    """
    Get document chunks for preview
    """
    try:
        # Get document info from DocumentStore
        doc = document_store.get_document(document_id, include_content=False)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # Get chunks from VectorStore (they have the chunk content and metadata)
        vector_chunks = vector_store.get_chunks_by_document_id(document_id)
        
        # Convert to response format
        chunks = []
        for chunk_metadata in vector_chunks:
            chunks.append(DocumentChunkInfo(
                chunk_id=chunk_metadata.chunk_id,
                content=chunk_metadata.content,
                page_number=chunk_metadata.page_number,
                start_char=chunk_metadata.start_char,
                end_char=chunk_metadata.end_char,
                chunk_index=chunk_metadata.chunk_metadata.get("chunk_index", 0),
                metadata=chunk_metadata.chunk_metadata
            ))
        
        # Sort chunks by index to maintain order
        chunks.sort(key=lambda x: x.chunk_index)
        
        response = DocumentChunksResponse(
            document_id=document_id,
            filename=doc.filename,
            chunks=chunks,
            total_chunks=len(chunks),
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document chunks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document chunks: {str(e)}"
        )


@router.get("/documents/stats", response_model=DocumentStoreStatsResponse)
async def get_document_store_stats() -> DocumentStoreStatsResponse:
    """
    Get document store statistics
    """
    try:
        stats = document_store.get_stats()
        
        response = DocumentStoreStatsResponse(
            total_documents=stats["total_documents"],
            total_versions=stats["total_versions"],
            total_size_bytes=stats["total_size_bytes"],
            db_size_mb=stats["db_size_mb"],
            last_updated=stats.get("last_updated")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get document store stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document store stats: {str(e)}"
        )


@router.post("/rag/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search documents using vector similarity
    """
    try:
        start_time = time.time()
        logger.info(f"Searching documents for query: '{request.query[:50]}...'")
        
        # Perform vector search
        results = vector_store.search(
            query=request.query,
            k=request.k,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids
        )
        
        # Convert to response format
        search_results = [
            SearchResult(
                chunk_id=result.chunk_metadata.chunk_id,
                document_id=result.chunk_metadata.document_id,
                content=result.chunk_metadata.content,
                similarity_score=result.similarity_score,
                rank=result.rank,
                page_number=result.chunk_metadata.page_number,
                metadata=result.chunk_metadata.chunk_metadata
            )
            for result in results
        ]
        
        search_time = time.time() - start_time
        
        response = SearchResponse(
            results=search_results,
            query=request.query,
            total_results=len(search_results),
            search_time=search_time
        )
        
        logger.info(f"Search completed: {len(search_results)} results in {search_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/chat/rag", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    """
    Chat with RAG-enhanced responses
    """
    try:
        logger.info(f"Received RAG chat request: '{request.message[:50]}...'")
        
        # Switch model if requested
        if request.model_name and request.model_name != chat_model_manager.model_name:
            logger.info(f"Switching to model: {request.model_name}")
            chat_model_manager.switch_model(request.model_name)
        
        # Ensure model is loaded
        if not chat_model_manager.is_loaded:
            logger.info("Model not loaded, loading now...")
            success = await chat_model_manager.load_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to load model"
                )
        
        # Get or create conversation
        conversation = None
        if request.conversation_id and request.conversation_id in active_conversations:
            conversation = active_conversations[request.conversation_id]
        else:
            # Create new conversation
            system_prompt = request.system_prompt or settings.default_system_prompt
            conversation = Conversation(
                system_prompt=system_prompt,
                max_length=settings.max_conversation_length
            )
            active_conversations[conversation.id] = conversation
        
        # Add user message
        conversation.add_user_message(request.message)
        
        # Prepare for RAG
        rag_used = False
        search_results = []
        context_length = 0
        enhanced_message = request.message
        
        # Use RAG if enabled and vector store has content
        if request.use_rag and vector_store.get_stats()["total_chunks"] > 0:
            try:
                # Get similar chunks for context
                results, formatted_context = vector_store.get_similar_chunks_for_rag(
                    query=request.message,
                    max_chunks=request.max_chunks,
                    similarity_threshold=request.similarity_threshold
                )
                
                if results and formatted_context:
                    # Enhance message with RAG context
                    enhanced_message = settings.rag.context_template.format(
                        context=formatted_context,
                        question=request.message
                    )
                    
                    # Convert results for response
                    search_results = [
                        SearchResult(
                            chunk_id=result.chunk_metadata.chunk_id,
                            document_id=result.chunk_metadata.document_id,
                            content=result.chunk_metadata.content,
                            similarity_score=result.similarity_score,
                            rank=result.rank,
                            page_number=result.chunk_metadata.page_number,
                            metadata=result.chunk_metadata.chunk_metadata
                        )
                        for result in results
                    ]
                    
                    rag_used = True
                    context_length = len(formatted_context)
                    
                    logger.info(f"RAG context added: {len(results)} chunks, {context_length} characters")
                    
            except Exception as e:
                logger.warning(f"RAG failed, falling back to normal chat: {e}")
        
        # Update the conversation with the enhanced message if RAG was used
        if rag_used:
            # Replace the last user message with enhanced version
            conversation.messages[-1].content = enhanced_message
        
        # Generate response
        result = await chat_model_manager.chat(
            conversation=conversation,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=False  # RAG responses are not streamed
        )
        
        # Update conversation in storage
        active_conversations[conversation.id] = result["conversation"]
        
        response = RAGChatResponse(
            response=result["response"],
            conversation_id=conversation.id,
            model_name=result["model_name"],
            generation_time=result["generation_time"],
            message_count=len(conversation.messages),
            parameters=result["parameters"],
            rag_used=rag_used,
            search_results=search_results if rag_used else None,
            context_length=context_length if rag_used else None
        )
        
        logger.info(f"RAG chat response generated in {result['generation_time']:.2f}s (RAG: {rag_used})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG chat failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG chat failed: {str(e)}"
        )


@router.get("/rag/stats", response_model=VectorStoreStatsResponse)
async def get_vector_store_stats() -> VectorStoreStatsResponse:
    """
    Get vector store statistics
    """
    try:
        stats = vector_store.get_stats()
        embeddings_info = embeddings_manager.get_model_info()
        
        response = VectorStoreStatsResponse(
            total_chunks=stats["total_chunks"],
            total_documents=stats["total_documents"],
            index_size_mb=stats["index_size_mb"],
            last_updated=stats.get("last_updated"),
            embeddings_model=embeddings_info["model_name"],
            embedding_dimension=embeddings_info.get("embedding_dimension")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get vector store stats: {str(e)}"
        )


@router.post("/rag/clear")
async def clear_vector_store() -> Dict[str, Any]:
    """
    Clear all data from the vector store
    """
    try:
        success = vector_store.clear()
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to clear vector store"
            )
        
        logger.info("Vector store cleared successfully")
        return {
            "message": "Vector store cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear vector store: {str(e)}"
        )