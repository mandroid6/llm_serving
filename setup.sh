#!/bin/bash

# LLM Serving API Setup Script
# This script creates a Python virtual environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up LLM Serving API..."
echo "=================================="

# Check if Python 3.8+ is available
python_cmd=""
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
        python_cmd="python3"
    fi
elif command -v python &> /dev/null; then
    python_version=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
        python_cmd="python"
    fi
fi

if [ -z "$python_cmd" ]; then
    echo "❌ Error: Python 3.8+ is required but not found."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

echo "✅ Found Python: $($python_cmd --version)"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

$python_cmd -m venv venv
echo "✅ Virtual environment created: ./venv"

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Check for GPU support
echo ""
echo "🔍 Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. You can enable GPU acceleration by setting:"
    echo "   export LLM_DEVICE=cuda"
    echo ""
    echo "💡 For optimal GPU performance, consider installing CUDA-optimized PyTorch:"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
else
    echo "ℹ️  No NVIDIA GPU detected. Using CPU mode (default)."
fi

# Create directories
echo ""
echo "📁 Creating required directories..."
mkdir -p models conversations logs
echo "✅ Directories created: ./models, ./conversations, ./logs"

# Create example .env file
echo ""
echo "⚙️  Creating example configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# LLM Serving API Configuration
LLM_MODEL_NAME=llama3-1b
LLM_DEVICE=cpu
LLM_MODEL_CACHE_DIR=./models
LLM_LOG_LEVEL=INFO
LLM_MAX_CONVERSATION_LENGTH=50
EOF
    echo "✅ Created .env file with default configuration"
else
    echo "ℹ️  .env file already exists, skipping creation"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "=================================="
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the API server:"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "3. In a new terminal, start the chat interface:"
echo "   source venv/bin/activate"
echo "   python chat_cli.py"
echo ""
echo "4. Or test the API directly:"
echo "   curl http://localhost:8000/api/v1/health"
echo ""
echo "🌐 Web interface will be available at:"
echo "   http://localhost:8000/docs"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "💡 Pro Tips:"
echo "   - Use 'deactivate' to exit the virtual environment"
echo "   - Modify .env file to customize settings"
echo "   - Use smaller models (gpt2, distilgpt2) if you have limited memory"
echo "   - Enable GPU with 'export LLM_DEVICE=cuda' for larger models"