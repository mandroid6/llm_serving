"""
Configuration settings for the LLM serving API with chat support
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from pydantic_settings import BaseSettings


class ModelProfile(BaseModel):
    """Configuration for a specific model"""
    name: str
    model_id: str  # Hugging Face model ID
    max_length: int = 2048
    chat_template: Optional[str] = None
    supports_chat: bool = False
    memory_gb: float = 4.0  # Estimated memory requirement
    description: str = ""
    
    # Model-specific generation defaults
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50
    default_max_tokens: int = 150


class ChatSettings(BaseModel):
    """Chat-specific configuration"""
    default_system_prompt: str = "You are a helpful AI assistant."
    max_conversation_length: int = 50  # Max turns to keep in context
    save_conversations: bool = True
    conversation_dir: str = "./conversations"
    auto_save_interval: int = 10  # Save every N messages
    streaming_enabled: bool = True
    show_typing_indicator: bool = True


class Settings(BaseSettings):
    """Application settings with chat support"""

    # API Configuration
    api_title: str = "LLM Serving API with Chat"
    api_description: str = "A FastAPI server for local LLM inference and chat"
    api_version: str = "1.1.0"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model Configuration
    current_model: str = "llama3-1b"  # Now using DialoGPT (no auth required)
    model_cache_dir: Optional[str] = "./models"
    
    # Chat Configuration
    chat: ChatSettings = Field(default_factory=ChatSettings)

    # Performance Settings
    torch_threads: int = 1
    device: str = "cpu"  # Start with CPU, can be changed to "cuda" if available

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"


# Define available model profiles
MODEL_PROFILES: Dict[str, ModelProfile] = {
    "gpt2": ModelProfile(
        name="GPT-2",
        model_id="gpt2",
        max_length=1024,
        supports_chat=False,
        memory_gb=2.0,
        description="OpenAI GPT-2 - Fast text generation",
        default_max_tokens=150
    ),
    
    "gpt2-medium": ModelProfile(
        name="GPT-2 Medium",
        model_id="gpt2-medium",
        max_length=1024,
        supports_chat=False,
        memory_gb=4.0,
        description="OpenAI GPT-2 Medium - Better quality",
        default_max_tokens=150
    ),
    
    "llama3-1b": ModelProfile(
        name="DialoGPT Medium (Chat)",
        model_id="microsoft/DialoGPT-medium",
        max_length=1024,
        chat_template="dialogpt",
        supports_chat=True,
        memory_gb=2.0,
        description="Microsoft DialoGPT Medium - Conversational AI",
        default_temperature=0.7,
        default_max_tokens=150
    ),
    
    "llama3-3b": ModelProfile(
        name="GPT-2 Large (Chat Mode)",
        model_id="gpt2-large",
        max_length=1024,
        chat_template="gpt2",
        supports_chat=True,
        memory_gb=3.0,
        description="GPT-2 Large configured for chat",
        default_temperature=0.7,
        default_max_tokens=150
    ),
    
    "distilgpt2": ModelProfile(
        name="DistilGPT-2",
        model_id="distilgpt2",
        max_length=1024,
        supports_chat=False,
        memory_gb=1.0,
        description="Distilled GPT-2 - Fastest option",
        default_max_tokens=100
    ),
}


# Chat templates for different models
CHAT_TEMPLATES: Dict[str, str] = {
    "llama3": """<|begin_of_text|>{% if messages[0]['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ messages[0]['content'] }}<|eot_id|>{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}""",

    "dialogpt": """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% if not loop.last %}<|endoftext|>{% endif %}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% if not loop.last %}<|endoftext|>{% endif %}{% endif %}{% endfor %}{% if add_generation_prompt %}<|endoftext|>{% endif %}""",
    
    "gpt2": """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}

{% elif message['role'] == 'user' %}Human: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}AI: {{ message['content'] }}

{% endif %}{% endfor %}{% if add_generation_prompt %}AI: {% endif %}""",
    
    "default": """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""
}


def get_model_profile(model_name: str) -> ModelProfile:
    """Get model profile by name"""
    if model_name not in MODEL_PROFILES:
        # Return a default profile for unknown models
        return ModelProfile(
            name=f"Custom ({model_name})",
            model_id=model_name,
            max_length=2048,
            supports_chat=False,
            memory_gb=4.0,
            description="Custom model profile"
        )
    return MODEL_PROFILES[model_name]


def get_chat_models() -> List[str]:
    """Get list of models that support chat"""
    return [name for name, profile in MODEL_PROFILES.items() if profile.supports_chat]


def get_all_models() -> List[str]:
    """Get list of all available models"""
    return list(MODEL_PROFILES.keys())


# Global settings instance
settings = Settings()
