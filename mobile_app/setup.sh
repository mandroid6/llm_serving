#!/bin/bash

# LLM Chat Mobile App Setup Script

echo "🚀 Setting up LLM Chat Mobile App..."

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the mobile_app directory"
    echo "   cd mobile_app && ./setup.sh"
    exit 1
fi

echo "📦 Installing dependencies..."
npm install

echo "🔍 Checking Expo CLI..."
if ! command -v expo &> /dev/null; then
    echo "📥 Installing Expo CLI globally..."
    npm install -g @expo/cli
else
    echo "✅ Expo CLI is already installed"
fi

echo "🎯 Creating placeholder assets..."
mkdir -p assets

# Create placeholder icon (this would normally be a proper PNG)
echo "📱 Note: You should add proper app icons to the assets/ directory:"
echo "   - icon.png (1024x1024)"
echo "   - adaptive-icon.png (1024x1024)"
echo "   - splash.png (1242x2436)"
echo "   - favicon.png (48x48)"

echo "🔧 Checking FastAPI server connection..."
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ FastAPI server is running and accessible"
else
    echo "⚠️  Warning: FastAPI server is not running on localhost:8000"
    echo "   Please start the server before running the mobile app:"
    echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
fi

echo ""
echo "🎉 Setup complete! To start the app:"
echo ""
echo "   npx expo start"
echo ""
echo "Then:"
echo "   - Press 'i' for iOS simulator"
echo "   - Press 'a' for Android emulator"
echo "   - Scan QR code with Expo Go app on your device"
echo ""
echo "📚 See README.md for more detailed instructions"