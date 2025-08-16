"""
Embeddings module for RAG functionality
Handles text embedding generation using sentence-transformers
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import time

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages text embeddings using sentence-transformers"""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.rag.embeddings_model
        self.device = device or settings.rag.embeddings_device

        # Auto-detect device if not specified
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: Optional[int] = None
        self.is_loaded = False
        self.load_time: Optional[float] = None

        logger.info(f"EmbeddingsManager initialized with model: {self.model_name}, device: {self.device}")

    def load_model(self) -> bool:
        """Load the sentence transformer model"""
        if self.is_loaded:
            logger.info("Embeddings model already loaded")
            return True

        try:
            start_time = time.time()
            logger.info(f"Loading embeddings model: {self.model_name}")

            # Load the model
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            self.load_time = time.time() - start_time
            self.is_loaded = True

            logger.info(f"Embeddings model loaded successfully in {self.load_time:.2f}s")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            logger.info(f"Model device: {self.device}")

            return True

        except Exception as e:
            logger.error(f"Failed to load embeddings model {self.model_name}: {e}")
            self.is_loaded = False
            return False
