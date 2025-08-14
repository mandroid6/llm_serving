"""
API endpoints for the LLM serving application
"""
import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging

from app.models.schemas import (
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from app.models.model_manager import model_manager

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

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
        if not model_manager.is_loaded:
            logger.info("Model not loaded, loading now...")
            success = await model_manager.load_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to load model"
                )
        
        # Generate text
        result = await model_manager.generate_text(
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
        model_info = model_manager.get_model_info()
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
        if model_manager.is_loaded:
            return {
                "message": "Model already loaded",
                "model_name": model_manager.model_name,
                "load_time": model_manager.load_time
            }
        
        logger.info("Loading model...")
        success = await model_manager.load_model()
        
        if success:
            return {
                "message": "Model loaded successfully",
                "model_name": model_manager.model_name,
                "load_time": model_manager.load_time
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