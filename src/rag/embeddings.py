"""
Embedding generation for RAG pipeline.

Tries to use SentenceTransformer (huggingface) when available. If the
sentence-transformers / huggingface_hub imports fail, we fall back to a
lightweight HashingVectorizer-based embedding with a fixed dimension. The
fallback is deterministic and avoids heavy external dependencies so the
project can run for demos and tests without the full HF stack.
"""
from typing import List
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import EMBEDDING_MODEL
from utils.logger import setup_logger

logger = setup_logger(__name__)

_USE_SENTENCE = False
_SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer
    _USE_SENTENCE = True
    _SentenceTransformer = SentenceTransformer
except Exception:
    # SentenceTransformers or huggingface_hub not available; will use fallback
    _USE_SENTENCE = False

_USE_HASHING = False
_HashingVectorizer = None
if not _USE_SENTENCE:
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        _USE_HASHING = True
        _HashingVectorizer = HashingVectorizer
    except Exception:
        _USE_HASHING = False


class EmbeddingGenerator:
    """Generate embeddings for text documents with a stable, fixed dimension.

    If SentenceTransformer is available it will be used. Otherwise a
    HashingVectorizer-based embedding with n_features=384 is used as a
    deterministic fallback.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, fallback_dim: int = 384):
        self.model_name = model_name
        self.fallback_dim = fallback_dim

        if _USE_SENTENCE and _SentenceTransformer is not None:
            try:
                # Try loading from local cache first to avoid network delays
                try:
                    self.model = _SentenceTransformer(model_name, local_files_only=True)
                except Exception:
                    logger.info(f"Loading SentenceTransformer model: {model_name}")
                    self.model = _SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                self._backend = "sentence"
                logger.info(f"Embedding dimension (sentence-transformers): {self.dimension}")
                return
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}); falling back")

        # Fallback: use HashingVectorizer to produce fixed-size vectors
        if _USE_HASHING and _HashingVectorizer is not None:
            logger.info(f"Using HashingVectorizer fallback with dim={self.fallback_dim}")
            # alternate_sign False keeps outputs non-signed which is more stable for FAISS
            self.vectorizer = _HashingVectorizer(n_features=self.fallback_dim, alternate_sign=False, norm='l2')
            self.dimension = self.fallback_dim
            self._backend = "hashing"
        else:
            # Last-resort fallback: use simple random but deterministic hashing
            logger.warning("No suitable embedding backend found; using deterministic hash fallback")
            self.dimension = self.fallback_dim
            self._backend = "hash_hash"

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if getattr(self, '_backend', None) == 'sentence':
            return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

        if getattr(self, '_backend', None) == 'hashing':
            vec = self.vectorizer.transform([text])
            return vec.toarray()[0]

        # deterministic hashing fallback
        arr = np.zeros(self.dimension, dtype=float)
        if not text:
            return arr
        # simple deterministic hash to spread tokens across dimension
        for i, ch in enumerate(text[:1024]):
            idx = (ord(ch) * (i + 1)) % self.dimension
            arr[idx] += (ord(ch) % 97) / 97.0
        # L2-normalize
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts (backend={getattr(self,'_backend',None)})...")

        if getattr(self, '_backend', None) == 'sentence':
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

        if getattr(self, '_backend', None) == 'hashing':
            vecs = self.vectorizer.transform(texts)
            arr = vecs.toarray()
            logger.info(f"Generated fallback embeddings with shape: {arr.shape}")
            return arr

        # hash_hash fallback
        arrs = [self.embed_text(t) for t in texts]
        return np.vstack(arrs)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        return self.embed_text(query)
