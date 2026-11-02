"""
Document retriever for RAG pipeline.
"""
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD
from utils.logger import setup_logger
from .embeddings import EmbeddingGenerator
from .vector_store import FAISSVectorStore

logger = setup_logger(__name__)

class MedicationKnowledgeRetriever:
    """Retrieve relevant medication knowledge for queries."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator = None, 
                 vector_store: FAISSVectorStore = None):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator: Optional EmbeddingGenerator instance
            vector_store: Optional FAISSVectorStore instance
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.vector_store = vector_store or FAISSVectorStore(
            dimension=self.embedding_generator.dimension
        )
        logger.info("Initialized MedicationKnowledgeRetriever")
    
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL, 
                 min_similarity: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of retrieved documents with metadata
        """
        logger.info(f"Retrieving documents for query: '{query[:100]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Filter by similarity threshold
        filtered_results = [
            {
                'content': doc,
                'similarity': score,
                'metadata': metadata
            }
            for doc, score, metadata in results
            if score >= min_similarity
        ]
        
        # If no documents pass the strict threshold, fallback to top retrieved documents
        if not filtered_results and results:
            filtered_results = [
                {
                    'content': doc,
                    'similarity': score,
                    'metadata': metadata
                }
                for doc, score, metadata in results
            ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents above threshold {min_similarity}")
        return filtered_results
    
    def add_knowledge_base(self, documents: List[str], metadata: List[Dict] = None):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        logger.info(f"Adding {len(documents)} documents to knowledge base...")
        
        # Generate embeddings
        embeddings = self.embedding_generator.embed_texts(documents)
        
        # Add to vector store
        self.vector_store.add_documents(embeddings, documents, metadata)
        
        logger.info("Successfully added documents to knowledge base")
    
    def save_knowledge_base(self, path: Path = None):
        """Save the knowledge base to disk."""
        self.vector_store.save(path)
    
    def load_knowledge_base(self, path: Path = None):
        """Load the knowledge base from disk."""
        self.vector_store.load(path)
    
    def get_context_for_query(self, query: str, max_context_length: int = 2000) -> str:
        """
        Get concatenated context for a query.
        
        Args:
            query: Search query
            max_context_length: Maximum total context length
            
        Returns:
            Concatenated context string
        """
        results = self.retrieve(query)
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            content = result['content']
            if current_length + len(content) > max_context_length:
                break
            context_parts.append(f"[Source {i+1}] {content}")
            current_length += len(content)
        
        context = "\n\n".join(context_parts)
        logger.info(f"Generated context of {len(context)} characters from {len(context_parts)} sources")
        
        return context
