#!/usr/bin/env python3
"""
Test script for voice integration with existing chat features
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.core.config import settings
    from app.voice import VoiceInputManager, VoiceInputMode
    from chat_cli import ChatInterface, ChatAPI
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_OK = False


async def test_voice_components():
    """Test basic voice components"""
    print("🧪 Testing Voice Components...")
    
    if not IMPORTS_OK:
        print("❌ Cannot test - import errors")
        return False
    
    try:
        # Test 1: Configuration
        print("  Testing configuration...")
        assert hasattr(settings, 'voice'), "Voice settings not found in config"
        assert settings.voice.enabled is not None, "Voice enabled setting not found"
        print("  ✅ Configuration test passed")
        
        # Test 2: VoiceInputManager creation
        print("  Testing VoiceInputManager creation...")
        voice_manager = VoiceInputManager(
            whisper_model=settings.voice.whisper_model,
            language=settings.voice.language,
            sample_rate=settings.voice.sample_rate,
            silence_threshold=settings.voice.silence_threshold,
            silence_duration=settings.voice.silence_duration,
            max_recording_time=settings.voice.max_recording_time,
            device_index=settings.voice.device_index
        )
        print("  ✅ VoiceInputManager creation test passed")
        
        # Test 3: Audio devices
        print("  Testing audio device detection...")
        devices = voice_manager.get_audio_devices()
        print(f"  📱 Found {len(devices)} audio devices")
        for device in devices[:3]:  # Show first 3 devices
            print(f"    - {device['name']} (Index: {device['index']})")
        print("  ✅ Audio device test passed")
        
        # Test 4: Whisper models
        print("  Testing Whisper model information...")
        models = voice_manager.get_whisper_models()
        print(f"  🎯 Available Whisper models: {list(models.keys())}")
        print("  ✅ Whisper model test passed")
        
        # Test 5: Status
        print("  Testing status information...")
        status = voice_manager.get_status()
        print(f"  📊 Voice input enabled: {status['is_enabled']}")
        print(f"  📊 Current model: {status['whisper_model']}")
        print(f"  📊 Language: {status['language']}")
        print("  ✅ Status test passed")
        
        print("✅ All voice component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Voice component test failed: {e}")
        return False


async def test_chat_interface_integration():
    """Test ChatInterface with voice integration"""
    print("🧪 Testing ChatInterface Integration...")
    
    if not IMPORTS_OK:
        print("❌ Cannot test - import errors")
        return False
    
    try:
        # Create a mock API for testing
        class MockChatAPI:
            def __init__(self):
                self.base_url = "http://localhost:8000"
            
            async def test_connection(self):
                return False  # Don't require actual server for this test
        
        # Test ChatInterface creation with voice
        print("  Testing ChatInterface creation...")
        api = MockChatAPI()
        chat_interface = ChatInterface(api)
        
        # Check voice integration
        print(f"  🎤 Voice available: {chat_interface.voice_enabled}")
        if chat_interface.voice_enabled:
            print(f"  🎤 Voice manager: {type(chat_interface.voice_manager).__name__}")
            print(f"  🎤 Voice mode: {chat_interface.voice_mode}")
        
        print("  ✅ ChatInterface integration test passed")
        
        # Test voice status panel
        if chat_interface.voice_enabled:
            print("  Testing voice status panel...")
            status_panel = chat_interface.get_voice_status_panel()
            if status_panel:
                print("  ✅ Voice status panel created successfully")
            else:
                print("  ⚠️ Voice status panel is None")
        
        print("✅ ChatInterface integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ChatInterface integration test failed: {e}")
        return False


async def test_configuration_integration():
    """Test configuration integration"""
    print("🧪 Testing Configuration Integration...")
    
    try:
        # Test voice settings structure
        voice_config = settings.voice
        
        required_fields = [
            'enabled', 'whisper_model', 'language', 'sample_rate',
            'silence_threshold', 'silence_duration', 'max_recording_time',
            'device_index', 'chunk_size', 'channels', 'show_transcription',
            'auto_send_transcription'
        ]
        
        for field in required_fields:
            assert hasattr(voice_config, field), f"Missing voice config field: {field}"
            print(f"  ✅ {field}: {getattr(voice_config, field)}")
        
        # Test environment variable handling
        print("  Testing environment variable prefix...")
        assert settings.Config.env_prefix == "LLM_", "Environment prefix should be LLM_"
        print("  ✅ Environment prefix test passed")
        
        print("✅ Configuration integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        return False


async def main():
    """Run all integration tests"""
    print("🚀 Starting Voice Integration Tests\n")
    
    tests = [
        ("Configuration Integration", test_configuration_integration),
        ("Voice Components", test_voice_components),
        ("ChatInterface Integration", test_chat_interface_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n💡 Voice integration is ready to use!")
        print("\n📚 Quick Start:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start server: uvicorn app.main:app --reload")
        print("3. Run chat: python chat_cli.py")
        print("4. Use /voice to toggle voice input")
        print("5. Use /record to record voice messages")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)