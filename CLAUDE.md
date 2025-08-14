# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is an LLM serving API built with FastAPI that serves GPT-2 for text generation. The architecture follows a layered pattern with clear separation of concerns:

- **API Layer** (`app/api/`): FastAPI endpoints with async handlers and Pydantic validation
- **Model Layer** (`app/models/`): Singleton ModelManager that handles Hugging Face transformer loading, caching, and inference
- **Core Layer** (`app/core/`): Configuration management with environment variables and structured logging
- **Testing**: Both API integration tests and standalone model tests

### Key Architectural Patterns

**Singleton Model Manager**: The `ModelManager` class in `app/models/model_manager.py` implements a singleton pattern for model caching. The model is loaded once and reused across requests. The `model_manager` global instance should be used throughout the application.

**Async-First Design**: All endpoints and model operations are async. The FastAPI app uses lifespan context managers for startup/shutdown. Model loading is asynchronous to avoid blocking the server.

**Environment-Based Configuration**: Settings in `app/core/config.py` use Pydantic Settings with `.env` file support and `LLM_` prefix for environment variables. Configuration is centralized through the global `settings` instance.

**Layered Error Handling**: Global exception handlers in `main.py` provide consistent error responses. Model operations include try-catch with detailed logging.

## Common Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Start development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run API integration tests (requires server to be running)
python client_test.py

# Interactive API testing
python client_test.py --interactive

# Standalone model testing (no server required)
python standalone_test.py

# Test with different models
python standalone_test.py --model gpt2-medium
python standalone_test.py --model distilgpt2
python standalone_test.py --model EleutherAI/gpt-neo-125M

# Interactive standalone testing
python standalone_test.py --interactive
python standalone_test.py --model gpt2-large --interactive

# Show available models
python standalone_test.py --list-models

# Unit tests with pytest
pytest tests/test_api.py
```

### Model Management

The first request to `/api/v1/generate` will automatically download and cache GPT-2 (~124MB) to `./models/`. To preload the model on server startup, uncomment the model loading lines in the lifespan function in `app/main.py`.

Use `/api/v1/load-model` endpoint to explicitly load the model without text generation.

## Configuration

Settings are managed through `app/core/config.py` and can be overridden via environment variables with `LLM_` prefix:

- `LLM_MODEL_NAME`: Model to use (default: "gpt2")
- `LLM_DEVICE`: "cpu" or "cuda" (default: "cpu")  
- `LLM_MODEL_CACHE_DIR`: Model cache directory (default: "./models")
- `LLM_LOG_LEVEL`: Logging level (default: "INFO")

## Key Files for Development

- `app/main.py`: FastAPI application setup, middleware, and global error handling
- `app/models/model_manager.py`: Core model loading and inference logic
- `app/api/endpoints.py`: API endpoint implementations
- `app/models/schemas.py`: Pydantic request/response models
- `standalone_test.py`: Direct model testing without API server

## API Endpoints

- `POST /api/v1/generate`: Text generation with configurable parameters
- `GET /api/v1/health`: Health check and uptime
- `GET /api/v1/model-info`: Model status and specifications  
- `POST /api/v1/load-model`: Explicit model loading
- `GET /docs`: Interactive API documentation