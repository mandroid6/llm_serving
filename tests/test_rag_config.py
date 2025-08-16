"""
Comprehensive tests for RAG (Retrieval-Augmented Generation) configuration settings

This test file covers Phase 1 Part 1 of the RAG implementation plan:
- RAG configuration settings validation
- Default values verification
- Environment variable integration
- Edge cases and validation boundaries
"""

import pytest
import os
import tempfile
from typing import List
from pydantic import ValidationError

from app.core.config import RAGSettings, Settings


class TestRAGSettings:
    """Test the RAGSettings configuration class"""

    def test_default_rag_settings(self):
        """Test that RAG settings have correct default values"""
        rag_settings = RAGSettings()
        
        # Document storage settings
        assert rag_settings.enabled is True
        assert rag_settings.documents_dir == "./documents"
        assert rag_settings.vector_db_dir == "./vector_db"
        assert rag_settings.max_file_size_mb == 50
        assert rag_settings.supported_formats == ["pdf", "txt", "md"]
        
        # Text processing settings
        assert rag_settings.chunk_size == 1000
        assert rag_settings.chunk_overlap == 200
        assert rag_settings.min_chunk_size == 100
        
        # Embeddings settings
        assert rag_settings.embeddings_model == "all-MiniLM-L6-v2"
        assert rag_settings.embeddings_device == "cpu"
        
        # Vector search settings
        assert rag_settings.similarity_threshold == 0.7
        assert rag_settings.max_chunks_per_query == 5
        assert rag_settings.rerank_chunks is True
        
        # RAG generation settings
        assert rag_settings.include_source_references is True
        assert rag_settings.max_context_length == 4000
        assert "Based on the following documents:" in rag_settings.context_template
        assert "{context}" in rag_settings.context_template
        assert "{question}" in rag_settings.context_template

    def test_rag_settings_validation(self):
        """Test validation of RAG settings values"""
        
        # Test valid settings
        valid_settings = RAGSettings(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            similarity_threshold=0.8,
            max_chunks_per_query=3
        )
        assert valid_settings.chunk_size == 500
        assert valid_settings.chunk_overlap == 100
        assert valid_settings.similarity_threshold == 0.8

    def test_chunk_settings_validation(self):
        """Test validation of chunk-related settings"""
        
        # Test that chunk_overlap should be less than chunk_size
        # This is a logical validation that could be added
        settings = RAGSettings(chunk_size=100, chunk_overlap=50)
        assert settings.chunk_overlap < settings.chunk_size
        
        # Test minimum chunk size
        settings = RAGSettings(min_chunk_size=10)
        assert settings.min_chunk_size == 10

    def test_similarity_threshold_boundaries(self):
        """Test similarity threshold boundary values"""
        
        # Test valid threshold values
        settings1 = RAGSettings(similarity_threshold=0.0)
        assert settings1.similarity_threshold == 0.0
        
        settings2 = RAGSettings(similarity_threshold=1.0)
        assert settings2.similarity_threshold == 1.0
        
        settings3 = RAGSettings(similarity_threshold=0.5)
        assert settings3.similarity_threshold == 0.5

    def test_supported_formats_validation(self):
        """Test supported file formats configuration"""
        
        # Test custom supported formats
        custom_formats = ["pdf", "txt", "docx", "html"]
        settings = RAGSettings(supported_formats=custom_formats)
        assert settings.supported_formats == custom_formats
        
        # Test empty formats list
        settings_empty = RAGSettings(supported_formats=[])
        assert settings_empty.supported_formats == []

    def test_embeddings_model_settings(self):
        """Test embeddings model configuration"""
        
        # Test custom embeddings model
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        settings = RAGSettings(embeddings_model=custom_model)
        assert settings.embeddings_model == custom_model
        
        # Test embeddings device settings
        settings_cuda = RAGSettings(embeddings_device="cuda")
        assert settings_cuda.embeddings_device == "cuda"
        
        settings_mps = RAGSettings(embeddings_device="mps")
        assert settings_mps.embeddings_device == "mps"

    def test_context_template_customization(self):
        """Test context template customization"""
        
        custom_template = "Context: {context}\n\nQuery: {question}\n\nResponse:"
        settings = RAGSettings(context_template=custom_template)
        assert settings.context_template == custom_template
        assert "{context}" in settings.context_template
        assert "{question}" in settings.context_template

    def test_max_chunks_validation(self):
        """Test max chunks per query validation"""
        
        # Test various valid values
        for max_chunks in [1, 3, 5, 10, 20]:
            settings = RAGSettings(max_chunks_per_query=max_chunks)
            assert settings.max_chunks_per_query == max_chunks

    def test_file_size_limits(self):
        """Test file size limit configuration"""
        
        # Test different file size limits
        for size_mb in [10, 25, 50, 100, 500]:
            settings = RAGSettings(max_file_size_mb=size_mb)
            assert settings.max_file_size_mb == size_mb


class TestSettingsRAGIntegration:
    """Test RAG settings integration with main Settings class"""

    def test_settings_includes_rag(self):
        """Test that Settings class includes RAG settings"""
        settings = Settings()
        
        # Verify RAG settings are included
        assert hasattr(settings, 'rag')
        assert isinstance(settings.rag, RAGSettings)
        
        # Verify default RAG settings are accessible
        assert settings.rag.enabled is True
        assert settings.rag.chunk_size == 1000
        assert settings.rag.embeddings_model == "all-MiniLM-L6-v2"

    def test_rag_settings_environment_variables(self):
        """Test RAG settings can be configured via environment variables"""
        
        # Note: This test would need environment variable setup
        # For now, we test the structure is correct for env var support
        settings = Settings()
        
        # Verify the Config class supports environment variables
        assert hasattr(settings, 'Config')
        assert settings.Config.env_prefix == "LLM_"
        assert settings.Config.env_file == ".env"

    def test_rag_disabled_scenario(self):
        """Test settings when RAG is disabled"""
        
        # Test explicitly disabled RAG
        disabled_rag = RAGSettings(enabled=False)
        assert disabled_rag.enabled is False
        
        # Other settings should still be valid
        assert disabled_rag.chunk_size == 1000
        assert disabled_rag.embeddings_model == "all-MiniLM-L6-v2"


class TestRAGDirectoryConfiguration:
    """Test RAG directory and path configuration"""

    def test_default_directories(self):
        """Test default directory configurations"""
        settings = RAGSettings()
        
        assert settings.documents_dir == "./documents"
        assert settings.vector_db_dir == "./vector_db"

    def test_custom_directories(self):
        """Test custom directory configurations"""
        
        custom_docs = "/custom/docs/path"
        custom_vector = "/custom/vector/path"
        
        settings = RAGSettings(
            documents_dir=custom_docs,
            vector_db_dir=custom_vector
        )
        
        assert settings.documents_dir == custom_docs
        assert settings.vector_db_dir == custom_vector

    def test_absolute_vs_relative_paths(self):
        """Test absolute and relative path handling"""
        
        # Relative paths
        rel_settings = RAGSettings(
            documents_dir="./data/docs",
            vector_db_dir="./data/vectors"
        )
        assert rel_settings.documents_dir == "./data/docs"
        assert rel_settings.vector_db_dir == "./data/vectors"
        
        # Absolute paths
        abs_settings = RAGSettings(
            documents_dir="/absolute/docs",
            vector_db_dir="/absolute/vectors"
        )
        assert abs_settings.documents_dir == "/absolute/docs"
        assert abs_settings.vector_db_dir == "/absolute/vectors"


class TestRAGPerformanceSettings:
    """Test performance-related RAG settings"""

    def test_chunk_size_performance_implications(self):
        """Test different chunk sizes and their implications"""
        
        # Small chunks
        small_settings = RAGSettings(chunk_size=200, chunk_overlap=50)
        assert small_settings.chunk_size == 200
        assert small_settings.chunk_overlap == 50
        
        # Large chunks
        large_settings = RAGSettings(chunk_size=2000, chunk_overlap=400)
        assert large_settings.chunk_size == 2000
        assert large_settings.chunk_overlap == 400

    def test_vector_search_performance_settings(self):
        """Test vector search performance settings"""
        
        # High precision settings
        high_precision = RAGSettings(
            similarity_threshold=0.9,
            max_chunks_per_query=10,
            rerank_chunks=True
        )
        assert high_precision.similarity_threshold == 0.9
        assert high_precision.max_chunks_per_query == 10
        assert high_precision.rerank_chunks is True
        
        # Fast search settings
        fast_search = RAGSettings(
            similarity_threshold=0.5,
            max_chunks_per_query=3,
            rerank_chunks=False
        )
        assert fast_search.similarity_threshold == 0.5
        assert fast_search.max_chunks_per_query == 3
        assert fast_search.rerank_chunks is False

    def test_context_length_settings(self):
        """Test context length configuration"""
        
        # Different context lengths
        for context_length in [1000, 2000, 4000, 8000]:
            settings = RAGSettings(max_context_length=context_length)
            assert settings.max_context_length == context_length


class TestRAGConfigurationEdgeCases:
    """Test edge cases and boundary conditions for RAG configuration"""

    def test_zero_values(self):
        """Test zero values where applicable"""
        
        # Zero chunk overlap (valid)
        settings = RAGSettings(chunk_overlap=0)
        assert settings.chunk_overlap == 0
        
        # Zero similarity threshold (valid)
        settings = RAGSettings(similarity_threshold=0.0)
        assert settings.similarity_threshold == 0.0

    def test_extreme_values(self):
        """Test extreme but valid values"""
        
        # Very large chunk size
        settings = RAGSettings(chunk_size=10000)
        assert settings.chunk_size == 10000
        
        # Very small chunk size
        settings = RAGSettings(chunk_size=50)
        assert settings.chunk_size == 50
        
        # Maximum similarity threshold
        settings = RAGSettings(similarity_threshold=1.0)
        assert settings.similarity_threshold == 1.0

    def test_empty_string_configurations(self):
        """Test empty string configurations"""
        
        # Empty embeddings model (should be allowed for custom handling)
        settings = RAGSettings(embeddings_model="")
        assert settings.embeddings_model == ""
        
        # Empty context template (should be allowed)
        settings = RAGSettings(context_template="")
        assert settings.context_template == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])