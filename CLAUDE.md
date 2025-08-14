# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an LLM serving API built with FastAPI that supports both GPT-2 and Llama3 models with conversational AI capabilities. The architecture follows a layered pattern with clear separation of concerns:

- **API Layer** (`app/api/`): FastAPI endpoints with async handlers and Pydantic validation for both text generation and chat
- **Chat Model Layer** (`app/models/`): ChatModelManager that handles multiple model types, conversation management, and context-aware responses  
- **Conversation Layer** (`app/models/`): Conversation and Message classes for chat history management with save/load functionality
- **Core Layer** (`app/core/`): Configuration management with model profiles, chat templates, and environment variables
- **CLI Interface**: Rich interactive chat interface with commands and model switching
- **Testing**: Comprehensive API integration tests and chat functionality tests

### Key Architectural Patterns

**ChatModelManager**: The `ChatModelManager` class in `app/models/chat_manager.py` implements a singleton pattern for model caching and conversation management. It supports both GPT-2 and Llama3 models with different chat templates. The `chat_model_manager` global instance should be used throughout the application.

**Conversation Context Management**: The `Conversation` class maintains message history with automatic trimming to token limits. Each conversation has a unique ID and supports JSON serialization for persistence.

**Model Profiles**: Pre-configured model profiles in `app/core/config.py` define model specifications, memory requirements, context lengths, and chat templates for each supported model.

**Async-First Design**: All endpoints and model operations are async. The FastAPI app uses lifespan context managers for startup/shutdown. Model loading is asynchronous to avoid blocking the server.

**Environment-Based Configuration**: Settings in `app/core/config.py` use Pydantic Settings with `.env` file support and `LLM_` prefix for environment variables. Configuration is centralized through the global `settings` instance.

**Rich CLI Interface**: The `chat_cli.py` provides a beautiful terminal interface using Rich and prompt-toolkit for interactive conversations with model switching and conversation management.

## Common Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start production server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Start interactive chat interface (in separate terminal)
python chat_cli.py
```

### Chat Interface Commands
```bash
# Available chat commands in chat_cli.py:
/help            # Show all available commands
/models          # List available models with status
/switch <model>  # Switch to a different model (e.g., /switch llama3-1b)
/save <name>     # Save current conversation to file
/load <name>     # Load a saved conversation
/clear           # Clear current conversation history
/quit            # Exit the chat interface

# Examples:
/switch llama3-3b         # Switch to Llama3 3B model
/save project_discussion  # Save conversation as project_discussion.json
/load project_discussion  # Load the saved conversation
```

### Testing
```bash
# Run API integration tests (requires server to be running)
python tests/client_test.py

# Interactive API testing
python tests/client_test.py --interactive

# Chat functionality tests (comprehensive)
pytest tests/test_chat.py -v

# All tests including chat functionality
pytest tests/ -v

# Standalone model testing (no server required)
python tests/standalone_test.py

# Test with different models
python tests/standalone_test.py --model gpt2-medium
python tests/standalone_test.py --model llama3-1b
python tests/standalone_test.py --model distilgpt2

# Interactive standalone testing
python tests/standalone_test.py --interactive
python tests/standalone_test.py --model llama3-1b --interactive

# Show available models
python tests/standalone_test.py --list-models
```

### Model Management

**Model Loading**: Models are automatically downloaded and cached to `./models/` on first use. Llama3 models are larger and may take longer to download:

- `gpt2`: ~124MB
- `gpt2-medium`: ~355MB  
- `llama3-1b`: ~1.2GB
- `llama3-3b`: ~3.2GB
- `distilgpt2`: ~82MB

**Default Models**: The system defaults to `llama3-1b` for chat functionality and `gpt2` for text generation.

**Model Switching**: Use the `/api/v1/chat/switch-model` endpoint or `/switch` command in CLI to change models during runtime.

**Memory Requirements**: 
- Llama3-1B: ~4GB RAM
- Llama3-3B: ~8GB RAM  
- GPT-2 models: ~2-4GB RAM

## Configuration

Settings are managed through `app/core/config.py` and can be overridden via environment variables with `LLM_` prefix:

### Core Settings
- `LLM_MODEL_NAME`: Default model (default: "llama3-1b")
- `LLM_DEVICE`: "cpu" or "cuda" (default: "cpu")  
- `LLM_MODEL_CACHE_DIR`: Model cache directory (default: "./models")
- `LLM_LOG_LEVEL`: Logging level (default: "INFO")

### Chat-Specific Settings  
- `LLM_MAX_CONVERSATION_LENGTH`: Maximum conversation turns (default: 50)
- `LLM_DEFAULT_SYSTEM_PROMPT`: Default system prompt for conversations
- `LLM_AUTO_SAVE_INTERVAL`: Auto-save conversations every N messages (default: 10)

### Model Profiles
Each model has a profile in `app/core/config.py` with:
- Model name and Hugging Face identifier
- Memory requirements and context length
- Chat template (for Llama3 models)
- Default generation parameters

## Key Files for Development

### Core Application Files
- `app/main.py`: FastAPI application setup, middleware, and global error handling
- `app/models/chat_manager.py`: ChatModelManager for conversation AI and model management
- `app/models/conversation.py`: Conversation and Message classes for chat history
- `app/api/endpoints.py`: API endpoint implementations (both chat and generation)
- `app/models/schemas.py`: Pydantic request/response models for all endpoints
- `app/core/config.py`: Configuration, model profiles, and chat templates

### Legacy Files (Still Used)
- `app/models/model_manager.py`: Legacy ModelManager for simple text generation
- `tests/standalone_test.py`: Direct model testing without API server
- `tests/client_test.py`: Basic API integration tests

### User Interfaces
- `chat_cli.py`: Interactive command-line chat interface with Rich formatting

### Testing
- `tests/test_api.py`: Basic API endpoint tests
- `tests/test_chat.py`: Comprehensive chat functionality tests

## API Endpoints

### Chat Endpoints (Primary)
- `POST /api/v1/chat`: Send a chat message with conversation context
- `POST /api/v1/chat/new`: Start a new conversation (returns conversation_id)
- `GET /api/v1/chat/models`: List all available models with status and descriptions
- `POST /api/v1/chat/switch-model`: Switch to a different model (affects all new conversations)
- `GET /api/v1/chat/conversation/{id}`: Retrieve conversation history by ID

### Legacy Generation Endpoints (Still Supported)
- `POST /api/v1/generate`: Text generation with configurable parameters (uses ModelManager)
- `GET /api/v1/health`: Health check and uptime
- `GET /api/v1/model-info`: Current model status and specifications  
- `POST /api/v1/load-model`: Explicit model loading
- `GET /docs`: Interactive API documentation

### Chat API Usage Patterns

**Starting a Conversation**:
1. Call `POST /api/v1/chat/new` to get a `conversation_id`
2. Use the `conversation_id` in subsequent `POST /api/v1/chat` requests
3. The system maintains context automatically within the conversation

**Model Switching**:
1. Call `GET /api/v1/chat/models` to see available models
2. Call `POST /api/v1/chat/switch-model` with desired model name
3. New conversations will use the new model

**Conversation Management**:
- Conversations are automatically trimmed when they exceed token limits
- Use `GET /api/v1/chat/conversation/{id}` to retrieve full conversation history
- Conversations persist in memory during server runtime

## Chat Templates and Model Behavior

### Llama3 Models
- Use proper Llama3 chat templates with special tokens
- Support system prompts and multi-turn conversations
- Optimized for chat and instruction-following

### GPT-2 Models  
- Use simple concatenation chat templates
- Better for text completion than conversation
- May require more explicit prompting for chat behavior

## Development Best Practices

### When Adding New Models
1. Add model profile to `MODEL_PROFILES` in `app/core/config.py`
2. Include appropriate chat template if the model supports it
3. Update `ChatModelManager._get_chat_template()` if special handling needed
4. Add tests in `tests/test_chat.py`
5. Update documentation

### When Modifying Chat Functionality
1. Update schemas in `app/models/schemas.py` for API changes
2. Modify `ChatModelManager` for core chat logic changes
3. Update `Conversation` class for history management changes
4. Test with both GPT-2 and Llama3 models
5. Update CLI interface if needed

### When Adding API Endpoints
1. Add endpoint to `app/api/endpoints.py`
2. Create appropriate Pydantic schemas
3. Add comprehensive tests in `tests/test_chat.py`
4. Update API documentation and examples