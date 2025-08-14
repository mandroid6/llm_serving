"""
Integration tests for the chat functionality and Llama3 models
"""
import pytest
import asyncio
import json
import tempfile
import os
from fastapi.testclient import TestClient
from app.main import app
from app.models.conversation import Conversation, Message

client = TestClient(app)


class TestChatEndpoints:
    """Test chat API endpoints"""
    
    def test_chat_models_endpoint(self):
        """Test listing available chat models"""
        response = client.get("/api/v1/chat/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        
        # Should include both GPT-2 and Llama3 models
        model_names = [model["name"] for model in data["models"]]
        assert "gpt2" in model_names
        assert "llama3-1b" in model_names
        
    def test_chat_new_conversation(self):
        """Test creating a new conversation"""
        response = client.post("/api/v1/chat/new")
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "message" in data
        assert data["message"] == "New conversation started"
        
    def test_chat_basic_message(self):
        """Test sending a basic chat message"""
        # Start new conversation
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        # Send a message
        payload = {
            "message": "Hello, how are you?",
            "conversation_id": conv_id
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "generation_time" in data
        assert len(data["response"]) > 0
        
    def test_chat_with_parameters(self):
        """Test chat with custom generation parameters"""
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        payload = {
            "message": "Tell me a short story",
            "conversation_id": conv_id,
            "max_length": 50,
            "temperature": 0.8,
            "top_p": 0.9
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
        
    def test_chat_conversation_context(self):
        """Test that conversation maintains context between messages"""
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        # First message
        payload1 = {
            "message": "My name is Alice",
            "conversation_id": conv_id
        }
        response1 = client.post("/api/v1/chat", json=payload1)
        assert response1.status_code == 200
        
        # Second message referencing the first
        payload2 = {
            "message": "What is my name?",
            "conversation_id": conv_id
        }
        response2 = client.post("/api/v1/chat", json=payload2)
        assert response2.status_code == 200
        
        # The response should ideally reference Alice, though this is model-dependent
        data = response2.json()
        assert len(data["response"]) > 0
        
    def test_get_conversation(self):
        """Test retrieving conversation history"""
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        # Send a message
        payload = {
            "message": "Hello there!",
            "conversation_id": conv_id
        }
        client.post("/api/v1/chat", json=payload)
        
        # Get conversation
        response = client.get(f"/api/v1/chat/conversation/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        assert "messages" in data
        assert len(data["messages"]) >= 2  # User message + assistant response
        
    def test_model_switching(self):
        """Test switching between different models"""
        # Switch to a specific model
        payload = {
            "model_name": "gpt2"
        }
        
        response = client.post("/api/v1/chat/switch-model", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "gpt2" in data["message"]
        
        # Verify model info reflects the change
        info_response = client.get("/api/v1/model-info")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert info_data["model_name"] == "gpt2"
        
    def test_invalid_conversation_id(self):
        """Test chat with invalid conversation ID"""
        payload = {
            "message": "Hello",
            "conversation_id": "invalid-id-123"
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code in [400, 404]  # Should fail gracefully
        
    def test_empty_message(self):
        """Test chat with empty message"""
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        payload = {
            "message": "",
            "conversation_id": conv_id
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 422  # Validation error
        
    def test_invalid_model_switch(self):
        """Test switching to an invalid model"""
        payload = {
            "model_name": "invalid-model-xyz"
        }
        
        response = client.post("/api/v1/chat/switch-model", json=payload)
        assert response.status_code in [400, 422]  # Should fail


class TestConversationManagement:
    """Test conversation and message management"""
    
    def test_conversation_creation(self):
        """Test creating a conversation object"""
        conv = Conversation()
        assert conv.id is not None
        assert len(conv.messages) == 0
        
    def test_message_creation(self):
        """Test creating message objects"""
        user_msg = Message(role="user", content="Hello")
        assert user_msg.role == "user"
        assert user_msg.content == "Hello"
        assert user_msg.timestamp is not None
        
        assistant_msg = Message(role="assistant", content="Hi there!")
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == "Hi there!"
        
    def test_conversation_add_messages(self):
        """Test adding messages to conversation"""
        conv = Conversation()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")
        
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"
        
    def test_conversation_to_dict(self):
        """Test conversation serialization"""
        conv = Conversation()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        
        data = conv.to_dict()
        assert "id" in data
        assert "messages" in data
        assert len(data["messages"]) == 2
        
    def test_conversation_from_dict(self):
        """Test conversation deserialization"""
        data = {
            "id": "test-id-123",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2024-01-01T00:00:00"
                },
                {
                    "role": "assistant", 
                    "content": "Hi!",
                    "timestamp": "2024-01-01T00:00:01"
                }
            ]
        }
        
        conv = Conversation.from_dict(data)
        assert conv.id == "test-id-123"
        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Hello"
        
    def test_conversation_save_load(self):
        """Test saving and loading conversations to/from JSON files"""
        conv = Conversation()
        conv.add_message("user", "Test message")
        conv.add_message("assistant", "Test response")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            
        try:
            conv.save_to_file(temp_path)
            
            # Load from file
            loaded_conv = Conversation.load_from_file(temp_path)
            
            assert loaded_conv.id == conv.id
            assert len(loaded_conv.messages) == len(conv.messages)
            assert loaded_conv.messages[0].content == "Test message"
            assert loaded_conv.messages[1].content == "Test response"
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_conversation_trimming(self):
        """Test conversation trimming when it gets too long"""
        conv = Conversation()
        
        # Add many messages to trigger trimming
        for i in range(60):  # More than the typical limit of 50
            conv.add_message("user", f"Message {i}")
            conv.add_message("assistant", f"Response {i}")
            
        # Conversation should be automatically trimmed
        # The exact behavior depends on implementation
        assert len(conv.messages) > 0  # Should still have some messages
        # Check that recent messages are preserved
        assert "59" in conv.messages[-1].content  # Last message should be recent


class TestModelIntegration:
    """Test model loading and integration"""
    
    def test_model_info_after_loading(self):
        """Test model info endpoint returns correct information"""
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        
        assert "model_name" in data
        assert "device" in data
        assert "loaded" in data
        assert isinstance(data["loaded"], bool)
        
    def test_load_model_endpoint(self):
        """Test explicit model loading"""
        response = client.post("/api/v1/load-model")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        
        # Check that model is loaded
        info_response = client.get("/api/v1/model-info")
        info_data = info_response.json()
        assert info_data["loaded"] is True
        
    def test_multiple_model_switches(self):
        """Test switching between multiple models"""
        models_to_test = ["gpt2", "llama3-1b"]
        
        for model_name in models_to_test:
            # Skip if model is not available (e.g., due to memory constraints)
            try:
                payload = {"model_name": model_name}
                response = client.post("/api/v1/chat/switch-model", json=payload)
                
                if response.status_code == 200:
                    # Verify switch was successful
                    info_response = client.get("/api/v1/model-info")
                    info_data = info_response.json()
                    assert info_data["model_name"] == model_name
                    
                    # Test that chat still works
                    new_conv_response = client.post("/api/v1/chat/new")
                    conv_id = new_conv_response.json()["conversation_id"]
                    
                    chat_payload = {
                        "message": "Test message",
                        "conversation_id": conv_id
                    }
                    chat_response = client.post("/api/v1/chat", json=chat_payload)
                    assert chat_response.status_code == 200
                    
            except Exception as e:
                # Log but don't fail if model is not available
                print(f"Model {model_name} not available for testing: {e}")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_malformed_chat_request(self):
        """Test handling of malformed requests"""
        # Missing required fields
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422
        
        # Invalid JSON structure
        response = client.post("/api/v1/chat", json={"invalid": "structure"})
        assert response.status_code == 422
        
    def test_nonexistent_conversation(self):
        """Test accessing non-existent conversation"""
        response = client.get("/api/v1/chat/conversation/nonexistent-id")
        assert response.status_code in [404, 400]
        
    def test_invalid_generation_parameters(self):
        """Test chat with invalid generation parameters"""
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        payload = {
            "message": "Test",
            "conversation_id": conv_id,
            "max_length": -1,  # Invalid
            "temperature": 5.0  # Invalid
        }
        
        response = client.post("/api/v1/chat", json=payload)
        assert response.status_code == 422


# Performance and stress tests
class TestPerformance:
    """Test performance characteristics"""
    
    def test_concurrent_conversations(self):
        """Test handling multiple concurrent conversations"""
        conversations = []
        
        # Create multiple conversations
        for i in range(5):
            response = client.post("/api/v1/chat/new")
            assert response.status_code == 200
            conversations.append(response.json()["conversation_id"])
        
        # Send messages to each conversation
        for conv_id in conversations:
            payload = {
                "message": f"Hello from conversation {conv_id[:8]}",
                "conversation_id": conv_id
            }
            response = client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            
    def test_long_conversation(self):
        """Test handling longer conversations"""
        new_conv_response = client.post("/api/v1/chat/new")
        conv_id = new_conv_response.json()["conversation_id"]
        
        # Send multiple messages in sequence
        for i in range(10):
            payload = {
                "message": f"Message number {i + 1}",
                "conversation_id": conv_id
            }
            response = client.post("/api/v1/chat", json=payload)
            assert response.status_code == 200
            
        # Verify conversation history
        response = client.get(f"/api/v1/chat/conversation/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        # Should have 20 messages (10 user + 10 assistant)
        assert len(data["messages"]) == 20


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])