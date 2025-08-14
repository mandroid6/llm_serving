# LLM Serving API with Chat Interface

## Overview
A production-ready LLM serving API built with FastAPI that supports both GPT-2 and Llama3 models with an interactive chat interface. Features include conversational AI, model switching, conversation management, and a rich command-line chat experience.

## Features
- 🤖 **Multiple Models**: GPT-2, GPT-2 Medium, Llama3 1B, Llama3 3B, DistilGPT2
- 💬 **Chat Interface**: Interactive CLI with rich formatting and commands
- 🔄 **Model Switching**: Switch between models during conversations
- 💾 **Conversation Management**: Save and load chat histories
- 🚀 **Fast API**: Async FastAPI server with comprehensive REST endpoints
- 🧠 **Context Awareness**: Maintains conversation history and context
- 🎨 **Rich CLI**: Beautiful terminal interface with colors and formatting

## Technical Stack
- **Framework**: FastAPI (async API server)
- **Models**: Hugging Face Transformers (GPT-2, Meta Llama3)
- **Chat System**: Custom conversation management with context
- **CLI**: Rich + prompt-toolkit for interactive experience
- **Runtime**: PyTorch with CPU/GPU support
- **Validation**: Pydantic schemas and request validation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Chat Interface
```bash
python chat_cli.py
```

### 4. Chat Commands
```bash
# List available models
/models

# Switch to Llama3 model
/switch llama3-1b

# Start chatting
Hello, how are you today?

# Save conversation
/save my_conversation

# Load previous conversation
/load my_conversation

# Get help
/help

# Exit
/quit
```

## Available Models

| Model | Size | Memory | Context | Best For |
|-------|------|---------|---------|----------|
| `gpt2` | 124MB | ~2GB | 1024 | Quick responses, lightweight |
| `gpt2-medium` | 355MB | ~4GB | 1024 | Better quality, moderate size |
| `llama3-1b` | 1.2GB | ~4GB | 8192 | Chat conversations, efficient |
| `llama3-3b` | 3.2GB | ~8GB | 8192 | High-quality chat, larger context |
| `distilgpt2` | 82MB | ~1GB | 1024 | Fastest, smallest model |

## Chat Interface Usage

### Basic Chat Example
```bash
$ python chat_cli.py

Welcome to LLM Chat Interface!
Current model: llama3-1b

You: Hello! Can you help me write a Python function?
Assistant: Of course! I'd be happy to help you write a Python function. 
What specific functionality would you like the function to have?

You: A function that calculates the factorial of a number
Assistant: Here's a Python function to calculate the factorial of a number:

```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
```

You: /save factorial_help
💾 Conversation saved to factorial_help.json

You: /quit
👋 Goodbye!
```

### Model Switching Example
```bash
You: /models
📋 Available Models:
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model         ┃ Status   ┃ Description                                  ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ llama3-1b     │ ✅ Loaded │ Meta Llama 3.2 1B Instruct                 │
│ llama3-3b     │ Available│ Meta Llama 3.2 3B Instruct                 │
│ gpt2          │ Available│ OpenAI GPT-2                                │
│ gpt2-medium   │ Available│ OpenAI GPT-2 Medium                         │
│ distilgpt2    │ Available│ DistilGPT2                                  │
└───────────────┴──────────┴──────────────────────────────────────────────┘

You: /switch gpt2-medium
🔄 Switching to model: gpt2-medium
⏳ Loading model... (this may take a moment)
✅ Successfully switched to gpt2-medium

You: Hello again with the new model!
Assistant: Hello! I'm now running on GPT-2 Medium. How can I assist you today?
```

## API Endpoints

### Chat Endpoints
- `POST /api/v1/chat` - Send a chat message
- `POST /api/v1/chat/new` - Start a new conversation
- `GET /api/v1/chat/models` - List available models
- `POST /api/v1/chat/switch-model` - Switch to a different model
- `GET /api/v1/chat/conversation/{id}` - Get conversation history

### Legacy Generation Endpoints
- `POST /api/v1/generate` - Generate text completion
- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/model-info` - Model information and status
- `POST /api/v1/load-model` - Load model explicitly

### API Usage Examples

#### Start a New Chat Conversation
```bash
curl -X POST "http://localhost:8000/api/v1/chat/new" \
  -H "Content-Type: application/json"
```

Response:
```json
{
  "conversation_id": "conv_abc123",
  "message": "New conversation started"
}
```

#### Send a Chat Message
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "conversation_id": "conv_abc123",
    "max_length": 100,
    "temperature": 0.7
  }'
```

Response:
```json
{
  "response": "The capital of France is Paris. It is located in the north-central part of the country.",
  "conversation_id": "conv_abc123",
  "generation_time": 1.23
}
```

#### Switch Models
```bash
curl -X POST "http://localhost:8000/api/v1/chat/switch-model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3-1b"}'
```

## Configuration

Environment variables with `LLM_` prefix:

- `LLM_MODEL_NAME`: Default model (default: "llama3-1b")
- `LLM_DEVICE`: "cpu" or "cuda" (default: "cpu")  
- `LLM_MODEL_CACHE_DIR`: Model cache directory (default: "./models")
- `LLM_LOG_LEVEL`: Logging level (default: "INFO")
- `LLM_MAX_CONVERSATION_LENGTH`: Max conversation turns (default: 50)

Example `.env` file:
```bash
LLM_MODEL_NAME=llama3-3b
LLM_DEVICE=cuda
LLM_MODEL_CACHE_DIR=/path/to/models
LLM_LOG_LEVEL=DEBUG
```

## Testing

### Run API Tests
```bash
# Basic API tests
python client_test.py

# Interactive API testing
python client_test.py --interactive

# Chat functionality tests
pytest tests/test_chat.py -v

# All tests
pytest tests/ -v
```

### Standalone Model Testing
```bash
# Test default model
python standalone_test.py

# Test specific model
python standalone_test.py --model llama3-1b

# Interactive mode
python standalone_test.py --interactive


## Project Structure

```
llm_serving/
├── README.md               # This file
├── CLAUDE.md              # Development guidelines
├── planning.md            # Implementation plan and status
├── requirements.txt       # Python dependencies
├── chat_cli.py            # Interactive chat interface
├── client_test.py         # API client tests
├── standalone_test.py     # Standalone model tests
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application setup
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py  # API endpoint implementations
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py     # Configuration and model profiles
│   │   └── logging.py    # Logging setup
│   └── models/
│       ├── __init__.py
│       ├── chat_manager.py    # ChatModelManager for conversation AI
│       ├── conversation.py    # Conversation and Message classes
│       ├── model_manager.py   # Legacy ModelManager (GPT-2 only)
│       └── schemas.py         # Pydantic request/response models
└── tests/
    ├── __init__.py
    ├── test_api.py        # Basic API tests
    └── test_chat.py       # Chat functionality tests
```

## Troubleshooting

### Common Issues

**Memory Issues**
- Start with smaller models (`gpt2`, `distilgpt2`, `llama3-1b`)
- Use `LLM_DEVICE=cpu` if GPU memory is insufficient
- Monitor system memory usage during model loading

**Model Loading Errors**
- Check internet connection for initial model downloads
- Verify sufficient disk space in model cache directory
- Clear model cache if corruption suspected: `rm -rf ./models/`

**Permission Errors**
- Ensure write permissions for model cache directory
- Use absolute paths in environment variables

**CLI Issues**
- Install required packages: `pip install rich prompt-toolkit`
- Use Python 3.8+ for best compatibility
- Check terminal supports UTF-8 for rich formatting

### Performance Tips

- **GPU Acceleration**: Set `LLM_DEVICE=cuda` if NVIDIA GPU available
- **Model Selection**: Choose model size based on available memory
- **Conversation Length**: Adjust `LLM_MAX_CONVERSATION_LENGTH` for memory optimization
- **Batch Processing**: Use API endpoints for multiple requests

## Development

### Adding New Models
1. Add model profile to `app/core/config.py`
2. Update `ChatModelManager` if special handling needed
3. Add tests in `tests/test_chat.py`
4. Update documentation

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- **Meta**: For the Llama3 models
- **OpenAI**: For GPT-2 models
- **Hugging Face**: For the transformers library
- **FastAPI**: For the excellent web framework
- **Rich**: For beautiful terminal formatting

---

*Happy chatting! 🤖✨*
