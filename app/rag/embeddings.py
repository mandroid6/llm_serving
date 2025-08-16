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
