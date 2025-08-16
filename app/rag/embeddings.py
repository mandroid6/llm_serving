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

    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string to embedding vector"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load embeddings model")

        try:
            # Convert to list for batch processing (even for single text)
            embeddings = self.model.encode(
                [text],
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            return embeddings[0]  # Return single embedding

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise RuntimeError(f"Text encoding failed: {e}")

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode a batch of texts to embedding vectors"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load embeddings model")

        if not texts:
            return np.array([])

        try:
            start_time = time.time()

            # Encode in batches to manage memory
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress and i == 0,  # Show progress only for first batch
                    convert_to_numpy=True
                )

                all_embeddings.append(batch_embeddings)

                if show_progress and i > 0:
                    progress = min((i + batch_size) / len(texts), 1.0)
                    logger.info(f"Encoding progress: {progress:.1%} ({i + len(batch)}/{len(texts)})")

            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)

            encoding_time = time.time() - start_time
            logger.info(f"Encoded {len(texts)} texts in {encoding_time:.2f}s ({len(texts)/encoding_time:.1f} texts/sec)")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise RuntimeError(f"Batch encoding failed: {e}")
