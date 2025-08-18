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
    default_system_prompt: str = "You are a helpful AI assistant. Provide clear, concise, and accurate responses. Do not repeat the user's question. Answer directly and be conversational."
    max_conversation_length: int = 50  # Max turns to keep in context
    save_conversations: bool = True
    conversation_dir: str = "./conversations"
    auto_save_interval: int = 10  # Save every N messages
    streaming_enabled: bool = True
    show_typing_indicator: bool = True


class VoiceSettings(BaseModel):
    """Voice input configuration"""
    enabled: bool = True
    whisper_model: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None  # None for auto-detection
    sample_rate: int = 16000
    silence_threshold: float = 0.01  # Audio level threshold for silence detection
    silence_duration: float = 2.0   # Seconds of silence before auto-stop
    max_recording_time: int = 60     # Maximum recording duration in seconds
    device_index: Optional[int] = None  # Specific audio device index (None for default)
    
    # Generation settings for voice input
    chunk_size: int = 1024  # Audio chunk size for processing
    channels: int = 1       # Audio channels (1 = mono)
    
    # UI settings
    show_transcription: bool = True    # Show transcribed text before sending
    auto_send_transcription: bool = True  # Automatically send transcribed text
    
    # Warning suppression (set to False to see Whisper warnings for debugging)
    suppress_warnings: bool = True


class RAGSettings(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration"""
    enabled: bool = True
    
    # Document storage settings
    documents_dir: str = "./documents"
    vector_db_dir: str = "./vector_db"
    document_store_db_path: str = "./document_store.db"  # DocumentStore database path
    max_file_size_mb: int = 50  # Maximum file size in MB
    supported_formats: List[str] = ["pdf", "txt", "md"]
    
    # Text processing settings
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size to keep
    
    # Embeddings settings
    embeddings_model: str = "all-MiniLM-L6-v2"  # Sentence-transformers model
    embeddings_device: str = "cpu"  # Device for embeddings computation
    
    # Vector search settings
    similarity_threshold: float = 0.7  # Minimum similarity for relevant chunks
    max_chunks_per_query: int = 5  # Maximum chunks to include in context
    rerank_chunks: bool = True  # Whether to rerank chunks by relevance
    
    # RAG generation settings
    include_source_references: bool = True  # Include source info in responses
    max_context_length: int = 4000  # Maximum context length for RAG
    context_template: str = "Based on the following documents:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"


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

    # Model Configuration (matching environment variable names)
    model_name: str = "qwen3-1.8b"  # Updated to use Qwen3 1.8B as default
    model_cache_dir: Optional[str] = "./models"
    
    # Chat Configuration (flattened to match environment variables)
    max_conversation_length: int = 50  # Moved from nested chat settings
    default_system_prompt: str = "You are a helpful AI assistant. Provide clear, concise, and accurate responses. Do not repeat the user's question. Answer directly and be conversational."
    auto_save_interval: int = 10  # Save every N messages
    
    # Chat settings (for backward compatibility)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    
    # Voice input settings
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    
    # RAG settings
    rag: RAGSettings = Field(default_factory=RAGSettings)

    # Performance Settings
    torch_threads: int = 1
    device: str = "cpu"  # Start with CPU, can be changed to "cuda" or "mps" if available

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"
        extra = "ignore"  # Ignore extra environment variables


# Define available model profiles
MODEL_PROFILES: Dict[str, ModelProfile] = {
    "gpt2": ModelProfile(
        name="GPT-2",
        model_id="gpt2",
        max_length=1024,
        chat_template="gpt2",
        supports_chat=True,  # Enable chat support
        memory_gb=2.0,
        description="OpenAI GPT-2 - Fast text generation with chat support",
        default_max_tokens=60,   # Shorter responses
        default_temperature=1.0, # Higher temperature for variety
        default_top_p=0.9,
        default_top_k=50
    ),
    
    "gpt2-medium": ModelProfile(
        name="GPT-2 Medium",
        model_id="gpt2-medium",
        max_length=1024,
        chat_template="gpt2",
        supports_chat=True,  # Enable chat support
        memory_gb=4.0,
        description="OpenAI GPT-2 Medium - Better quality with chat support",
        default_max_tokens=80,
        default_temperature=0.9,
        default_top_p=0.95,
        default_top_k=50
    ),
    
    "llama3-1b": ModelProfile(
        name="DialoGPT Medium (Chat)",
        model_id="microsoft/DialoGPT-medium",
        max_length=1024,
        chat_template="dialogpt",
        supports_chat=True,
        memory_gb=2.0,
        description="Microsoft DialoGPT Medium - Conversational AI",
        default_temperature=0.9,  # Higher for more variety
        default_max_tokens=60,    # Shorter for better responses
        default_top_p=0.9,
        default_top_k=50
    ),
    
    "llama3-3b": ModelProfile(
        name="GPT-2 Large (Chat Mode)",
        model_id="gpt2-large",
        max_length=1024,
        chat_template="gpt2",
        supports_chat=True,
        memory_gb=3.0,
        description="GPT-2 Large configured for chat",
        default_temperature=0.8,
        default_max_tokens=80,    # Moderate length
        default_top_p=0.95,
        default_top_k=50
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
    
    # Qwen3 Models (larger, more capable models)
    "qwen3-1.8b": ModelProfile(
        name="Qwen2.5 1.8B Instruct",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_length=32768,  # Much longer context
        chat_template="qwen",
        supports_chat=True,
        memory_gb=6.0,
        description="Qwen2.5 1.8B - High-quality multilingual chat model",
        default_temperature=0.7,
        default_max_tokens=200,
        default_top_p=0.8,
        default_top_k=20
    ),
    
    "qwen3-3b": ModelProfile(
        name="Qwen2.5 3B Instruct", 
        model_id="Qwen/Qwen2.5-3B-Instruct",
        max_length=32768,
        chat_template="qwen",
        supports_chat=True,
        memory_gb=8.0,
        description="Qwen2.5 3B - Advanced multilingual chat model",
        default_temperature=0.7,
        default_max_tokens=300,
        default_top_p=0.8,
        default_top_k=20
    ),
    
    "qwen3-7b": ModelProfile(
        name="Qwen2.5 7B Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct", 
        max_length=32768,
        chat_template="qwen",
        supports_chat=True,
        memory_gb=16.0,
        description="Qwen2.5 7B - State-of-the-art multilingual chat model",
        default_temperature=0.7,
        default_max_tokens=400,
        default_top_p=0.8,
        default_top_k=20
    ),
    
    "qwen3-14b": ModelProfile(
        name="Qwen2.5 14B Instruct",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        max_length=32768,
        chat_template="qwen", 
        supports_chat=True,
        memory_gb=32.0,
        description="Qwen2.5 14B - Premium multilingual chat model (requires high memory)",
        default_temperature=0.7,
        default_max_tokens=500,
        default_top_p=0.8,
        default_top_k=20
    ),
    
    # DeepSeek Models (CPU-compatible versions)
    "deepseek-coder-1.3b": ModelProfile(
        name="DeepSeek Coder 1.3B",
        model_id="deepseek-ai/deepseek-coder-1.3b-instruct",
        max_length=16384,
        chat_template="deepseek",
        supports_chat=True,
        memory_gb=6.0,
        description="DeepSeek Coder 1.3B - Code generation and reasoning (CPU compatible)",
        default_temperature=0.7,
        default_max_tokens=300,
        default_top_p=0.8,
        default_top_k=20
    ),
    
    "deepseek-coder-6.7b": ModelProfile(
        name="DeepSeek Coder 6.7B", 
        model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        max_length=16384,
        chat_template="deepseek",
        supports_chat=True,
        memory_gb=16.0,
        description="DeepSeek Coder 6.7B - Advanced code generation (CPU compatible)",
        default_temperature=0.7,
        default_max_tokens=400,
        default_top_p=0.8,
        default_top_k=20
    ),
    
    "deepseek-math-7b": ModelProfile(
        name="DeepSeek Math 7B",
        model_id="deepseek-ai/deepseek-math-7b-instruct",
        max_length=4096,
        chat_template="deepseek", 
        supports_chat=True,
        memory_gb=16.0,
        description="DeepSeek Math 7B - Mathematical reasoning and problem solving",
        default_temperature=0.7,
        default_max_tokens=400,
        default_top_p=0.8,
        default_top_k=20
    ),
}


# Chat templates for different models
CHAT_TEMPLATES: Dict[str, str] = {
    "llama3": """<|begin_of_text|>{% if messages[0]['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ messages[0]['content'] }}<|eot_id|>{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}""",

    "qwen": """<|im_start|>system
{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}{% else %}You are a helpful assistant.{% endif %}<|im_end|>
{% for message in messages %}{% if message['role'] != 'system' %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}""",

    "deepseek": """<|begin_of_text|>{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}""",

    "dialogpt": """{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% if not loop.last %}<|endoftext|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|endoftext|>{% endif %}""",
    
    "gpt2": """{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}

{% elif message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Bot: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Bot:{% endif %}""",
    
    "default": """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}Human: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""
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


def get_optimal_device() -> str:
    """Get the optimal device for the current hardware"""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Metal Performance Shaders on Mac
    else:
        return "cpu"


# Global settings instance
settings = Settings()

# Auto-detect optimal device if using default
if settings.device == "cpu":
    optimal_device = get_optimal_device()
    if optimal_device != "cpu":
        settings.device = optimal_device
