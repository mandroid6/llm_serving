"""
Main FastAPI application
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.endpoints import router
from app.models.model_manager import model_manager

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for the FastAPI application
    """
    # Startup
    logger.info("Starting LLM Serving API...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")
    
    # Optionally preload the model on startup
    # Uncomment the next lines if you want to preload the model
    # logger.info("Preloading model...")
    # await model_manager.load_model()
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Serving API...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "LLM Serving API",
        "version": settings.api_version,
        "model": settings.model_name,
        "endpoints": {
            "generate": "/api/v1/generate",
            "health": "/api/v1/health",
            "model_info": "/api/v1/model-info",
            "load_model": "/api/v1/load-model"
        },
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """
    Global HTTP exception handler
    """
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """
    Global exception handler for unhandled exceptions
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )