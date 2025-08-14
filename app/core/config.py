"""
Configuration settings for the LLM serving API
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    api_title: str = "LLM Serving API"
    api_description: str = "A FastAPI server for local LLM inference"
    api_version: str = "1.0.0"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model Configuration
    model_name: str = "gpt2"  # GPT-2 (closest open alternative to GPT-3)
    model_cache_dir: Optional[str] = "./models"
    max_model_length: int = 1024  # GPT-2 supports up to 1024 tokens

    # Generation Defaults
    default_max_length: int = 150  # Increased for GPT-2's capabilities
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50

    # Performance Settings
    torch_threads: int = 1
    device: str = "cpu"  # Start with CPU, can be changed to "cuda" if available

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"


# Global settings instance
settings = Settings()
