"""
FAISS vector store for medication knowledge retrieval.
"""
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import VECTOR_DB_DIR, TOP_K_RETRIEVAL
from utils.logger import setup_logger

logger = setup_logger(__name__)

class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents = []
        self.metadata = []
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on specified type."""
        if self.index_type == "Flat":
            # Exact search using L2 distance
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created FAISS index: {self.index_type} with dimension {self.dimension}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: Optional[List[dict]] = None):
        """
        Add documents to the vector store.
        
        Args:
            embeddings: Document embeddings (N x dimension)
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Train index if needed (for IVF)
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))
        
        logger.info(f"Added {len(documents)} documents to vector store. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (document, similarity_score, metadata) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty!")
            return []
        
        # Reshape query for FAISS
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                # Convert L2 distance to similarity score (inverse)
                similarity = 1 / (1 + dist)
                results.append((
                    self.documents[idx],
                    float(similarity),
                    self.metadata[idx]
                ))
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def save(self, path: Optional[Path] = None):
        """Save vector store to disk."""
        save_path = path or VECTOR_DB_DIR
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save documents and metadata
        with open(save_path / "documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        logger.info(f"Saved vector store to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """Load vector store from disk."""
        load_path = path or VECTOR_DB_DIR
        load_path = Path(load_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load documents and metadata
        with open(load_path / "documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        logger.info(f"Loaded vector store from {load_path} with {self.index.ntotal} documents")
    
    def clear(self):
        """Clear all documents from the vector store."""
        self._create_index()
        self.documents = []
        self.metadata = []
        logger.info("Cleared vector store")
