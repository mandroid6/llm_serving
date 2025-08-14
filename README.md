# LLM Serving API with Chat Interface

## Overview
A production-ready LLM serving API built with FastAPI that supports both GPT-2 and Llama3 models with an interactive chat interface. Features include conversational AI, model switching, conversation management, and a rich command-line chat experience.

## Features
- 🤖 **Multiple Models**: GPT-2, GPT-2 Medium, Qwen3 1.8B, Qwen3 3B, Llama3 models, DistilGPT2
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

## Installation & Setup

### 🚀 Automated Setup (Recommended)

**Quick setup with our installation script:**

```bash
# Clone or download the project
git clone <repository-url> llm_serving
cd llm_serving

# Run the automated setup script
./setup.sh
```

The setup script will:
- ✅ Check Python 3.8+ installation
- 🐍 Create a virtual environment (`./venv`)
- 📦 Install all dependencies from requirements.txt
- 📁 Create required directories (`models`, `conversations`, `logs`)
- ⚙️ Create default `.env` configuration file
- 🖥️ Check for GPU support and provide optimization tips

**After setup completion:**

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. In a new terminal, start chat interface
source venv/bin/activate
python chat_cli.py
```

### 🛠️ Manual Setup

**For advanced users or custom installations:**

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create required directories
mkdir -p models conversations logs

# 4. Create .env file (optional)
cp .env.example .env  # Edit as needed

# 5. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 📋 Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB+ RAM (4GB+ recommended for larger models)
- **Storage**: 5GB+ free space for model downloads
- **GPU** (optional): NVIDIA GPU with CUDA for acceleration

## Quick Start

**Once installation is complete, here's how to use the system:**

### Chat Commands
```bash
# List available models
/models

# Switch to Qwen model 
/switch qwen3-1.8b

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

### Basic Models (No Authentication Required)
| Model | Size | Memory | Context | Best For |
|-------|------|---------|---------|----------|
| `gpt2` | 124MB | ~2GB | 1024 | Quick responses, lightweight |
| `gpt2-medium` | 355MB | ~4GB | 1024 | Better quality, moderate size |
| `llama3-1b` | 1.2GB | ~4GB | 8192 | Chat conversations, efficient |
| `llama3-3b` | 3.2GB | ~8GB | 8192 | High-quality chat, larger context |
| `distilgpt2` | 82MB | ~1GB | 1024 | Fastest, smallest model |

### Advanced Models (Qwen Series - High Performance)
| Model | Size | Memory | Context | Best For |
|-------|------|---------|---------|----------|
| `qwen3-1.8b` | ~3.5GB | ~6GB | 32768 | ✨ **Advanced multilingual chat** |
| `qwen3-3b` | ~6GB | ~8GB | 32768 | 🚀 **High-quality reasoning** |
| `qwen3-7b` | ~14GB | ~16GB | 32768 | 🎯 **Professional tasks** |
| `qwen3-14b` | ~28GB | ~32GB | 32768 | 🏆 **Enterprise-grade AI** |

**💡 Tip**: Qwen models offer superior performance but require more memory. Start with `qwen3-1.8b` for the best balance of quality and resource usage.

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
│ llama3-1b     │ ✅ Loaded │ DialoGPT Medium (Chat)                      │
│ qwen3-1.8b    │ Available│ Qwen2.5 1.8B - Advanced multilingual chat  │
│ qwen3-3b      │ Available│ Qwen2.5 3B - High-quality reasoning         │
│ qwen3-7b      │ Available│ Qwen2.5 7B - Professional tasks             │
│ gpt2          │ Available│ OpenAI GPT-2                                │
│ gpt2-medium   │ Available│ OpenAI GPT-2 Medium                         │
│ distilgpt2    │ Available│ DistilGPT2                                  │
└───────────────┴──────────┴──────────────────────────────────────────────┘

You: /switch qwen3-1.8b
🔄 Switching to model: qwen3-1.8b
⏳ Loading model... (this may take a moment)
✅ Successfully switched to Qwen2.5 1.8B Instruct

You: Hello! Can you help me write a Python function?
Assistant: Hello! I'd be happy to help you write a Python function. Could you tell me what specific functionality you need? For example:

- Mathematical calculations
- Data processing
- File operations
- Web scraping
- API interactions

Just describe what you want the function to do, and I'll create it for you with proper documentation and examples.

You: /switch qwen3-7b  # For more complex tasks
🔄 Switching to model: qwen3-7b
⏳ Loading model... (downloading ~14GB, please wait)
✅ Successfully switched to Qwen2.5 7B Instruct

You: Write a complete web scraper for e-commerce prices
Assistant: I'll create a comprehensive web scraper for e-commerce price monitoring. Here's a complete solution...
[Much more detailed and sophisticated response]
```

### GPU Configuration for Larger Models

**🎯 Enable GPU Support**
For Qwen 7B+ models, GPU acceleration is strongly recommended:

```bash
# Check if you have NVIDIA GPU
nvidia-smi

# Install GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Enable GPU in environment
export LLM_DEVICE=cuda

# Start server with GPU
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**💾 Memory Requirements by Model**
- **qwen3-1.8b**: 6GB RAM (or 4GB VRAM)
- **qwen3-3b**: 8GB RAM (or 6GB VRAM)
- **qwen3-7b**: 16GB RAM (or 8GB VRAM) ⚡ **GPU Recommended**
- **qwen3-14b**: 32GB RAM (or 16GB VRAM) 🚨 **GPU Required**

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

- `LLM_MODEL_NAME`: Default model (default: "qwen3-1.8b")
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
python tests/client_test.py

# Interactive API testing
python tests/client_test.py --interactive

# Chat functionality tests
pytest tests/test_chat.py -v

# All tests
pytest tests/ -v
```

### Standalone Model Testing
```bash
# Test default model
python tests/standalone_test.py

# Test specific model
python tests/standalone_test.py --model llama3-1b

# Interactive mode
python tests/standalone_test.py --interactive
```

## Project Structure

```text
llm_serving/
├── README.md               # This file
├── CLAUDE.md              # Development guidelines
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
├── setup.sh               # Automated setup script
├── chat_cli.py            # Interactive chat interface
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
    ├── client_test.py     # API client tests
    ├── standalone_test.py # Standalone model tests
    ├── test_api.py        # Basic API tests
    └── test_chat.py       # Chat functionality tests
```

## Troubleshooting

### Model Authentication Issues

**🔒 "Access to model is restricted" Error**
Some models (like Meta Llama) require Hugging Face authentication:

```bash
# Option 1: Use pre-configured open models (recommended)
# The system is already configured with open alternatives:
# - llama3-1b → DialoGPT Medium (no auth required)
# - llama3-3b → GPT-2 Large (no auth required)

# Option 2: Set up Hugging Face authentication
pip install huggingface_hub
huggingface-cli login
# Then accept model license at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

### Chat Behavior Issues

**🔄 Model Returns Same Message as Prompt**
When models echo your input instead of responding:

```bash
# Test with higher temperature
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "temperature": 0.9,
    "max_tokens": 100
  }'

# Or adjust in chat CLI
💬 You: /switch gpt2
💬 You: Hello, how are you?
```

**🎲 GPT-2 Returns Random Text**
GPT-2 needs proper conversation context:
- ✅ **Fixed**: GPT-2 now has improved chat templates
- Use `/switch gpt2` for better conversational responses
- If still having issues, try `/switch qwen3-1.8b` (Qwen model)

**🔧 Model Parameter Tuning**
Adjust generation parameters for better responses:

| Parameter | Low Value Issue | High Value Issue | Recommended |
|-----------|----------------|------------------|-------------|
| `temperature` | Repetitive, boring | Random, nonsensical | 0.7-0.9 |
| `max_tokens` | Cut-off responses | Too verbose | 50-150 |
| `top_p` | Limited variety | Too scattered | 0.8-0.95 |
| `top_k` | Repetitive | Unfocused | 40-60 |

### Chat Interface Issues

**⚡ "asyncio.run() cannot be called from a running event loop"**
Running chat CLI from async environment (Jupyter, IPython):

```bash
# Solution 1: Install nest-asyncio
pip install nest-asyncio
python chat_cli.py

# Solution 2: Use regular terminal
# Open plain command prompt/terminal (not Jupyter)
python chat_cli.py

# Solution 3: Use API directly
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Solution 4: Use web interface
# Open browser: http://localhost:8000/docs
```

### Model Performance Issues

**🚫 503 Service Unavailable on Model Switch**
Model failed to load, usually due to memory:

```bash
# Check available memory
free -h  # Linux
vm_stat | grep "free\|inactive"  # macOS

# Solutions:
💬 You: /switch qwen3-1.8b    # Try default model (6GB)
💬 You: /switch gpt2         # Even smaller (2GB)
💬 You: /switch distilgpt2   # Smallest (1GB)

# Or enable GPU if available
export LLM_DEVICE=cuda
```

**📊 Model Memory Requirements**

| Model | Memory | Speed | Quality | Best For |
|-------|--------|-------|---------|----------|
| `distilgpt2` | ~1GB | ⚡⚡⚡ | ⭐⭐ | Testing, low-resource |
| `gpt2` | ~2GB | ⚡⚡ | ⭐⭐⭐ | ✅ **General chat** |
| `llama3-1b` | ~2GB | ⚡⚡ | ⭐⭐⭐⭐ | ✅ **Best conversation** |
| `llama3-3b` | ~3GB | ⚡ | ⭐⭐⭐⭐⭐ | High-quality chat |

### Common Issues

**💾 Memory Issues**
- Start with smaller models (`gpt2`, `distilgpt2`, `llama3-1b`)
- Use `LLM_DEVICE=cpu` if GPU memory is insufficient
- Monitor system memory usage during model loading
- Close other applications to free up RAM

**📥 Model Loading Errors**
- Check internet connection for initial model downloads
- Verify sufficient disk space in model cache directory (models can be 1-3GB)
- Clear model cache if corruption suspected: `rm -rf ./models/`
- Wait patiently - first download can take several minutes

**🔐 Permission Errors**
- Ensure write permissions for model cache directory
- Use absolute paths in environment variables
- On Linux/macOS: `chmod 755 ./models/`

**🖥️ CLI Issues**
- Install required packages: `pip install rich prompt-toolkit nest-asyncio`
- Use Python 3.8+ for best compatibility
- Check terminal supports UTF-8 for rich formatting
- Try different terminal if colors/formatting broken

### API Testing & Debugging

**🧪 Test API Endpoints**
```bash
# 1. Health check
curl http://localhost:8000/api/v1/health

# 2. List models
curl http://localhost:8000/api/v1/chat/models

# 3. Test chat
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# 4. Test with custom parameters
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a Python function to add two numbers",
    "temperature": 0.8,
    "max_tokens": 150
  }'
```

**🔍 Check Server Logs**
Monitor the uvicorn server output for detailed error messages:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Watch for error messages in the console output
```

### Getting Help

If you're still having issues:

1. **📋 Check server logs** for detailed error messages
2. **🧪 Test with API directly** using curl commands above
3. **💾 Try different models** starting with smallest (`distilgpt2`)
4. **🔄 Restart the server** after configuration changes
5. **🌐 Use web interface** at `http://localhost:8000/docs` for testing

**🎯 Quick Diagnosis**
```bash
# Run this diagnostic sequence:
curl http://localhost:8000/api/v1/health && echo "✅ Server OK" || echo "❌ Server issue"
curl http://localhost:8000/api/v1/chat/models && echo "✅ Models OK" || echo "❌ Model config issue"
curl -X POST "http://localhost:8000/api/v1/chat" -H "Content-Type: application/json" -d '{"message":"test"}' && echo "✅ Chat OK" || echo "❌ Chat issue"
```

### Performance Tips

- **GPU Acceleration**: Set `LLM_DEVICE=cuda` if NVIDIA GPU available
- **Model Selection**: Choose model size based on available memory
- **Conversation Length**: Adjust `LLM_MAX_CONVERSATION_LENGTH` for memory optimization
- **Batch Processing**: Use API endpoints for multiple requests

## Development

### Adding New Models (e.g., Qwen3, Claude, GPT-4)

**🚀 Complete Guide to Adding Large Language Models**

#### Step 1: Add Model Profile
Edit `app/core/config.py` and add your model to `MODEL_PROFILES`:

```python
"your-model-name": ModelProfile(
    name="Display Name",
    model_id="huggingface/model-id",  # e.g., "Qwen/Qwen2.5-7B-Instruct"
    max_length=32768,                 # Context window size
    chat_template="qwen",             # Template type (see step 2)
    supports_chat=True,
    memory_gb=16.0,                   # Estimated memory requirement
    description="Model description",
    default_temperature=0.7,
    default_max_tokens=400,
    default_top_p=0.8,
    default_top_k=20
),
```

#### Step 2: Add Chat Template (if needed)
If your model uses a unique format, add to `CHAT_TEMPLATES`:

```python
"your-template": """<|start|>system
{{ system_message }}<|end|>
{% for message in messages %}<|start|>{{ message['role'] }}
{{ message['content'] }}<|end|>
{% endfor %}<|start|>assistant
""",
```

**Common Template Formats:**
- **Qwen**: `<|im_start|>role\ncontent<|im_end|>`
- **Llama**: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>`
- **ChatML**: `<|im_start|>role\ncontent<|im_end|>`
- **Alpaca**: `### Human:\ncontent\n\n### Assistant:\n`

#### Step 3: Update Model Handling
Add your model ID to the force list in `app/models/chat_manager.py`:

```python
force_custom_template = [
    # ... existing models
    "your-org/your-model-name",
]
```

#### Step 4: Test Your Model
```bash
# Restart server
uvicorn app.main:app --reload

# Test via API
curl -X POST "http://localhost:8000/api/v1/chat/switch-model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "your-model-name"}'

# Test in CLI
python chat_cli.py
# /switch your-model-name
# Hello! Test message
```

#### Step 5: Performance Optimization

**For Large Models (7B+):**
```python
# Add to model profile
default_temperature=0.7,        # Lower for consistency
default_max_tokens=300,         # Reasonable length
default_top_p=0.8,             # Focused sampling
default_top_k=20,              # Reduced options
```

**Memory Management:**
```python
# In generation config (chat_manager.py)
generation_config = GenerationConfig(
    repetition_penalty=1.1,     # Reduce repetition
    no_repeat_ngram_size=3,     # Avoid n-gram repetition
    early_stopping=True,        # Stop at natural points
    use_cache=True,            # Enable KV caching
)
```

### Real Example: Adding Mistral 7B

```python
# 1. Add to MODEL_PROFILES
"mistral-7b": ModelProfile(
    name="Mistral 7B Instruct",
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_length=32768,
    chat_template="mistral",
    supports_chat=True,
    memory_gb=16.0,
    description="Mistral 7B - High-performance instruct model",
    default_temperature=0.7,
    default_max_tokens=300
),

# 2. Add template to CHAT_TEMPLATES
"mistral": """<s>{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}""",

# 3. Add to force_custom_template list
"mistralai/Mistral-7B-Instruct-v0.2",
```

### Troubleshooting New Models

**❌ Model won't load**
- Check Hugging Face model ID is correct
- Verify you have enough memory/VRAM
- Check if model requires authentication

**❌ Generates poor responses**
- Adjust temperature (0.1-1.5)
- Modify chat template format
- Check max_tokens setting
- Add repetition penalties

**❌ Template errors**
- Validate Jinja2 syntax
- Check message role handling
- Test with simple templates first

### Model Recommendations by Use Case

| Model Type | Size | Use Case | Config Tips |
|------------|------|----------|-------------|
| **Code** | 7B+ | Programming tasks | `temperature=0.1`, longer tokens |
| **Chat** | 3-7B | Conversations | `temperature=0.7`, moderate tokens |
| **Creative** | 7B+ | Writing, stories | `temperature=1.0`, high tokens |
| **Analysis** | 14B+ | Complex reasoning | `temperature=0.3`, very long context |

### Advanced Configuration

**Quantization Support:**
```python
# For models with quantized versions
model_id="microsoft/DialoGPT-medium-int8",  # 8-bit version
memory_gb=1.0,  # Reduced memory requirement
```

**Multi-GPU Setup:**
```python
# Set device mapping for large models
device_map="auto",  # Automatic GPU distribution
```

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
