"""
Basic tests for the LLM Serving API
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "uptime" in data


def test_model_info_endpoint():
    """Test the model info endpoint"""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "device" in data
    assert "loaded" in data


def test_generate_endpoint():
    """Test the text generation endpoint"""
    payload = {
        "prompt": "Hello world",
        "max_length": 20,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "prompt" in data
    assert "generation_time" in data


def test_generate_invalid_prompt():
    """Test generation with invalid prompt"""
    payload = {
        "prompt": "",  # Empty prompt should fail validation
        "max_length": 20
    }
    
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422  # Validation error


def test_generate_invalid_parameters():
    """Test generation with invalid parameters"""
    payload = {
        "prompt": "Test prompt",
        "max_length": -1,  # Invalid max_length
        "temperature": 3.0  # Invalid temperature
    }
    
    response = client.post("/api/v1/generate", json=payload)
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__])