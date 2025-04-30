import os
from functools import lru_cache
from typing import List, Optional


class Embedder:
    """Handles text embedding with multiple backends"""

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        provider: str = "sbert",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        endpoint: Optional[str] = None,
        cache_size: int = 1024,
    ):
        self.model_name = model_name_or_path
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_base = api_base
        self.endpoint = endpoint

        # Try to get API key from environment if not provided
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "azure":
                self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            elif self.provider == "huggingface":
                self.api_key = os.environ.get("HF_API_KEY")
            elif self.provider == "cohere":
                self.api_key = os.environ.get("COHERE_API_KEY")

        # Try to get endpoint from environment if not provided
        if self.endpoint is None:
            if self.provider == "sbert":
                self.endpoint = os.environ.get("SBERT_ENDPOINT")
            elif self.provider == "openai":
                self.endpoint = os.environ.get("OPENAI_ENDPOINT")
            elif self.provider == "azure":
                self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            elif self.provider == "huggingface":
                self.endpoint = os.environ.get("HF_ENDPOINT")
            elif self.provider == "cohere":
                self.endpoint = os.environ.get("COHERE_ENDPOINT")

        self._model = None

        # Initialize model based on provider
        if self.provider == "sbert":
            try:
                from sentence_transformers import SentenceTransformer

                if self.endpoint:
                    # If endpoint is provided, use it for remote inference
                    self._model = SentenceTransformer(
                        model_name_or_path, endpoint=self.endpoint
                    )
                else:
                    # Otherwise use local model
                    self._model = SentenceTransformer(model_name_or_path)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is required for SBERT embeddings. "
                    "Install with: pip install sentence-transformers"
                )

    @lru_cache(maxsize=1024)
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for input text"""
        if self.provider == "sbert":
            return self._sbert_embed(text)
        elif self.provider == "openai":
            return self._openai_embed(text)
        elif self.provider == "azure":
            return self._azure_embed(text)
        elif self.provider == "huggingface":
            return self._hf_embed(text)
        elif self.provider == "cohere":
            return self._cohere_embed(text)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    # Implement provider-specific embedding methods
    def _sbert_embed(self, text: str) -> List[float]:
        vector = self._model.encode(text)
        return vector.tolist() if hasattr(vector, "tolist") else list(vector)

    def _openai_embed(self, text: str) -> List[float]:
        # Implementation for OpenAI embeddings
        pass

    # Implement other provider methods similarly
