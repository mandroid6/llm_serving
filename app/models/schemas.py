"""
Pydantic models for API request/response schemas
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator


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