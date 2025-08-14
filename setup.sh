#!/bin/bash

# LLM Serving API Setup Script
# This script creates a Python virtual environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up LLM Serving API..."
echo "=================================="

# Function to compare version numbers
version_ge() {
    # Returns 0 (true) if $1 >= $2, 1 (false) otherwise
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check if Python 3.8+ is available
python_cmd=""
min_version="3.8"

if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    if version_ge "$python_version" "$min_version"; then
        python_cmd="python3"
    fi
elif command -v python &> /dev/null; then
    python_version=$(python --version 2>&1 | awk '{print $2}')
    if version_ge "$python_version" "$min_version"; then
        python_cmd="python"
    fi
fi

# Fallback: check for specific Python versions if the above didn't work
if [ -z "$python_cmd" ]; then
    for py_ver in python3.12 python3.11 python3.10 python3.9 python3.8; do
        if command -v "$py_ver" &> /dev/null; then
            python_cmd="$py_ver"
            python_version=$($py_ver --version 2>&1 | awk '{print $2}')
            break
        fi
    done
fi

if [ -z "$python_cmd" ]; then
    echo "❌ Error: Python 3.8+ is required but not found."
    echo ""
    echo "🍎 On macOS, you can install Python using:"
    echo "   • Homebrew: brew install python"
    echo "   • Official installer: https://www.python.org/downloads/"
    echo "   • pyenv: pyenv install 3.11.0 && pyenv global 3.11.0"
    echo ""
    exit 1
fi

echo "✅ Found Python: $($python_cmd --version)"

# Install system dependencies
echo ""
echo "🔧 Installing system dependencies..."

# Check if we're on macOS
if [[ $(uname -s) == "Darwin" ]]; then
    echo "🍎 macOS detected - installing system dependencies with Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "📦 Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for current session
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            # Apple Silicon Mac
            export PATH="/opt/homebrew/bin:$PATH"
            echo "✅ Added Homebrew to PATH (Apple Silicon)"
        elif [[ -f "/usr/local/bin/brew" ]]; then
            # Intel Mac
            export PATH="/usr/local/bin:$PATH"
            echo "✅ Added Homebrew to PATH (Intel)"
        fi
    else
        echo "✅ Homebrew already installed"
    fi
    
    # Update Homebrew
    echo "🔄 Updating Homebrew..."
    brew update
    
    # Install FFmpeg (required for Whisper audio processing)
    echo "🎵 Installing FFmpeg for voice input support..."
    if brew list ffmpeg &>/dev/null; then
        echo "✅ FFmpeg already installed"
    else
        brew install ffmpeg
        echo "✅ FFmpeg installed successfully"
    fi
    
    # Install PortAudio (required for PyAudio)
    echo "🎤 Installing PortAudio for voice recording..."
    if brew list portaudio &>/dev/null; then
        echo "✅ PortAudio already installed"
    else
        brew install portaudio
        echo "✅ PortAudio installed successfully"
    fi
    
    echo "✅ macOS system dependencies installed"
    
elif [[ $(uname -s) == "Linux" ]]; then
    echo "🐧 Linux detected - please install system dependencies manually:"
    echo ""
    echo "   For Ubuntu/Debian:"
    echo "   sudo apt update"
    echo "   sudo apt install ffmpeg portaudio19-dev python3-dev"
    echo ""
    echo "   For CentOS/RHEL/Fedora:"
    echo "   sudo yum install ffmpeg portaudio-devel python3-devel"
    echo "   # or: sudo dnf install ffmpeg portaudio-devel python3-devel"
    echo ""
    echo "⚠️  Voice input may not work without these dependencies!"
    
else
    echo "❓ Unknown operating system. Please install these dependencies manually:"
    echo "   - FFmpeg (for audio processing)"
    echo "   - PortAudio (for audio recording)"
    echo ""
    echo "⚠️  Voice input may not work without these dependencies!"
fi

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

# Check for GPU support (including Apple Silicon)
echo ""
echo "🔍 Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. You can enable GPU acceleration by setting:"
    echo "   export LLM_DEVICE=cuda"
    echo ""
    echo "💡 For optimal GPU performance, consider installing CUDA-optimized PyTorch:"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
elif [[ $(uname -m) == "arm64" ]] && [[ $(uname -s) == "Darwin" ]]; then
    echo "🍎 Apple Silicon Mac detected. You can enable Metal GPU acceleration by setting:"
    echo "   export LLM_DEVICE=mps"
    echo ""
    echo "💡 For optimal performance on Apple Silicon, ensure you have the latest PyTorch:"
    echo "   pip install --upgrade torch torchvision torchaudio"
elif [[ $(uname -s) == "Darwin" ]]; then
    echo "🍎 Intel Mac detected. Using CPU mode (default)."
    echo "💡 Consider upgrading to Apple Silicon for better ML performance."
else
    echo "ℹ️  No GPU acceleration detected. Using CPU mode (default)."
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
    # Detect if we're on Apple Silicon for default device setting
    if [[ $(uname -m) == "arm64" ]] && [[ $(uname -s) == "Darwin" ]]; then
        default_device="mps"
    else
        default_device="cpu"
    fi

    cat > .env << EOF
# LLM Serving API Configuration
LLM_MODEL_NAME=llama3-1b
LLM_DEVICE=${default_device}
LLM_MODEL_CACHE_DIR=./models
LLM_LOG_LEVEL=INFO
LLM_MAX_CONVERSATION_LENGTH=50

# Voice Input Configuration
LLM_VOICE_ENABLED=true
LLM_VOICE_WHISPER_MODEL=base
LLM_VOICE_SUPPRESS_WARNINGS=true
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
echo "🎤 Voice Input Features:"
echo "   • Use '/voice' to toggle voice input mode"
echo "   • Use '/record' to record voice messages"
echo "   • Use '/voice-settings' to configure voice options"
echo "   • Use '/devices' to list audio devices"
echo "   • All transcription happens locally (offline)"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "💡 Pro Tips:"
echo "   - Use 'deactivate' to exit the virtual environment"
echo "   - Modify .env file to customize settings"
echo "   - Use smaller models (gpt2, distilgpt2) if you have limited memory"
echo "   - Voice input works offline using OpenAI Whisper"
echo "   - Set LLM_VOICE_SUPPRESS_WARNINGS=false to see Whisper warnings for debugging"
if [[ $(uname -m) == "arm64" ]] && [[ $(uname -s) == "Darwin" ]]; then
    echo "   - Enable Apple Silicon GPU with 'export LLM_DEVICE=mps' for better performance"
else
    echo "   - Enable GPU with 'export LLM_DEVICE=cuda' for larger models (if NVIDIA GPU available)"
fi
