#!/bin/bash
# Setup script for LLM Serving API using uv

set -e

echo "🚀 Setting up LLM Serving API with uv"
echo "=================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    pip install uv
fi

echo "🔧 Creating virtual environment..."
uv venv

echo "📁 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "📦 Installing dependencies..."
uv pip install -e .

echo "🛠️ Installing development dependencies..."
uv pip install -e ".[dev]"

echo "✅ Setup complete!"
echo ""
echo "To activate the environment manually:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "To run the server:"
echo "  uvicorn app.main:app --reload"
echo ""
echo "To test standalone:"
echo "  python standalone_test.py --interactive"