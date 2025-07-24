"""
Vector Store Module for RAG Assistant
Handles embeddings generation and vector database operations using FAISS and ChromaDB
"""
import logging
import pickle
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from text_chunker import TextChunk
import faiss
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformer models
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 cache_dir: Optional[Path] = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the SentenceTransformer model
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(
                model_name, 
                cache_folder=str(cache_dir) if cache_dir else None
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        try:
            # Generate embeddings in batches to handle memory better
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text
            
        Returns:
            NumPy array of the query embedding
        """
        return self.model.encode([query], convert_to_numpy=True)[0]


class FAISSVectorStore:
    """
    FAISS-based vector store for similarity search
    """
    
    def __init__(self, embedding_dim: int, index_path: Optional[Path] = None):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_path: Path to save/load the index
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index = None
        self.chunks = []  # Store original chunks
        self.metadata = []  # Store metadata
        
        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[TextChunk]):
        """
        Add embeddings and chunks to the index
        
        Args:
            embeddings: NumPy array of embeddings
            chunks: List of corresponding TextChunk objects
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Normalize embeddings for better similarity search
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.metadata.append({
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'token_count': chunk.token_count
            })
        
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity score (higher is better)
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity))
        
        return results
    
    def save(self, path: Optional[Path] = None):
        """Save the index and metadata"""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path specified for saving")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save chunks and metadata
        with open(save_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved FAISS index to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """Load the index and metadata"""
        load_path = path or self.index_path
        if not load_path or not (load_path / "index.faiss").exists():
            logger.warning("No saved index found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(load_path / "index.faiss"))
            
            # Load chunks and metadata
            with open(load_path / "chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            with open(load_path / "metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index from {load_path} with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


class ChromaDBVectorStore:
    """
    ChromaDB-based vector store for similarity search
    """
    
    def __init__(self, collection_name: str = "rag_documents", 
                 persist_directory: Optional[Path] = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        settings = Settings()
        if persist_directory:
            settings = Settings(persist_directory=str(persist_directory))
        
        self.client = chromadb.Client(settings)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(collection_name)
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[TextChunk]):
        """
        Add embeddings and chunks to ChromaDB
        
        Args:
            embeddings: NumPy array of embeddings
            chunks: List of corresponding TextChunk objects
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Prepare data for ChromaDB
        ids = [f"{chunk.source}_{chunk.chunk_id}" for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index,
                'token_count': chunk.token_count or 0
            }
            for chunk in chunks
        ]
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(embeddings)} embeddings to ChromaDB collection")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks in ChromaDB
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            chunks_with_scores = []
            for i in range(len(results['ids'][0])):
                # Reconstruct TextChunk from ChromaDB results
                metadata = results['metadatas'][0][i]
                chunk = TextChunk(
                    content=results['documents'][0][i],
                    source=metadata['source'],
                    chunk_id=metadata['chunk_id'],
                    start_index=metadata['start_index'],
                    end_index=metadata['end_index'],
                    token_count=metadata.get('token_count')
                )
                
                # ChromaDB returns distances, convert to similarity
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)
                
                chunks_with_scores.append((chunk, similarity))
            
            return chunks_with_scores
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []


class VectorStore:
    """
    Unified vector store interface that can use either FAISS or ChromaDB
    """
    
    def __init__(self, 
                 store_type: str = "FAISS",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: Optional[Path] = None,
                 model_cache_dir: Optional[Path] = None):
        """
        Initialize the vector store
        
        Args:
            store_type: Type of vector store ("FAISS" or "ChromaDB")
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist the store
            model_cache_dir: Directory to cache the embedding model
        """
        self.store_type = store_type
        self.persist_directory = persist_directory
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            embedding_model, 
            cache_dir=model_cache_dir
        )
        
        # Initialize appropriate vector store
        if store_type.upper() == "FAISS":
            self.vector_store = FAISSVectorStore(
                self.embedding_generator.embedding_dim,
                persist_directory
            )
        elif store_type.upper() == "CHROMADB":
            self.vector_store = ChromaDBVectorStore(
                persist_directory=persist_directory
            )
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
        
        logger.info(f"Initialized {store_type} vector store")
    
    def add_chunks(self, chunks: List[TextChunk]):
        """
        Add text chunks to the vector store
        
        Args:
            chunks: List of TextChunk objects
        """
        if not chunks:
            logger.warning("No chunks provided")
            return
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, chunks)
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search in vector store
        return self.vector_store.search(query_embedding, k)
    
    def save(self):
        """Save the vector store"""
        if hasattr(self.vector_store, 'save'):
            self.vector_store.save()
    
    def load(self) -> bool:
        """Load the vector store"""
        if hasattr(self.vector_store, 'load'):
            return self.vector_store.load()
        return False


if __name__ == "__main__":
    from config import VECTOR_DB_DIR, EMBEDDINGS_MODEL_CACHE, EMBEDDING_MODEL_NAME
    
    # Create sample chunks for testing
    sample_chunks = [
        TextChunk("Machine learning is a subset of AI", "test.txt", 0, 0, 50, 12),
        TextChunk("Python is a programming language", "test.txt", 1, 51, 100, 8),
        TextChunk("Data science involves statistics and programming", "test.txt", 2, 101, 150, 11),
    ]
    
    # Test FAISS vector store
    print("=== Testing FAISS Vector Store ===")
    faiss_store = VectorStore(
        store_type="FAISS",
        embedding_model=EMBEDDING_MODEL_NAME,
        persist_directory=VECTOR_DB_DIR / "faiss",
        model_cache_dir=EMBEDDINGS_MODEL_CACHE
    )
    
    faiss_store.add_chunks(sample_chunks)
    results = faiss_store.search("What is machine learning?", k=2)
    
    for chunk, score in results:
        print(f"Score: {score:.3f} | {chunk.content[:50]}...")
    
    # Test ChromaDB vector store
    print("\n=== Testing ChromaDB Vector Store ===")
    chroma_store = VectorStore(
        store_type="ChromaDB",
        embedding_model=EMBEDDING_MODEL_NAME,
        persist_directory=VECTOR_DB_DIR / "chroma",
        model_cache_dir=EMBEDDINGS_MODEL_CACHE
    )
    
    chroma_store.add_chunks(sample_chunks)
    results = chroma_store.search("programming languages", k=2)
    
    for chunk, score in results:
        print(f"Score: {score:.3f} | {chunk.content[:50]}...")
