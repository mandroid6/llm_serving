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
    ChatMessage
)
from app.models.chat_manager import ChatModelManager
from app.models.conversation import Conversation
from app.core.config import settings, get_model_profile, get_chat_models, get_all_models

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Global chat model manager instance
chat_model_manager = ChatModelManager()

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
            system_prompt = request.system_prompt or settings.chat.default_system_prompt
            conversation = Conversation(
                system_prompt=system_prompt,
                max_length=settings.chat.max_conversation_length
            )
            active_conversations[conversation.id] = conversation
        
        # Add user message
        conversation.add_user_message(request.message)
        
        # Generate response
        result = None
        async for chunk in chat_model_manager.chat(
            conversation=conversation,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=request.stream
        ):
            if chunk.get("finished", False):
                result = chunk
                break
            # For streaming, we would handle chunks here
        
        if result is None:
            raise RuntimeError("Failed to get response from chat model")
        
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
            max_length=request.max_length or settings.chat.max_conversation_length
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