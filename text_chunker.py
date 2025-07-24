"""
Text Chunking Module for RAG Assistant
Handles text splitting using LangChain's RecursiveCharacterTextSplitter
"""
import logging
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    Represents a chunk of text with metadata
    """
    content: str
    source: str
    chunk_id: int
    start_index: int
    end_index: int
    token_count: Optional[int] = None


class TextChunker:
    """
    Handles text chunking with configurable parameters
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: Custom separators for splitting (default uses common separators)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use default separators if none provided
        if separators is None:
            separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                " ",     # Spaces
                "",      # Characters
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
    def chunk_text(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Split text into chunks
        
        Args:
            text: Text to be chunked
            source: Source identifier (e.g., filename)
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            logger.warning(f"Empty text provided for source: {source}")
            return []
        
        # Split text using LangChain's splitter
        chunks = self.text_splitter.split_text(text)
        
        text_chunks = []
        current_position = 0
        
        for i, chunk_content in enumerate(chunks):
            # Find the actual position of this chunk in the original text
            chunk_start = text.find(chunk_content, current_position)
            if chunk_start == -1:
                # Fallback if exact match not found
                chunk_start = current_position
            
            chunk_end = chunk_start + len(chunk_content)
            
            text_chunk = TextChunk(
                content=chunk_content.strip(),
                source=source,
                chunk_id=i,
                start_index=chunk_start,
                end_index=chunk_end,
                token_count=self.estimate_token_count(chunk_content)
            )
            
            text_chunks.append(text_chunk)
            current_position = chunk_end
        
        logger.info(f"Split {source} into {len(text_chunks)} chunks")
        return text_chunks
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[TextChunk]:
        """
        Chunk multiple documents
        
        Args:
            documents: Dictionary mapping document names to their text content
            
        Returns:
            List of all text chunks from all documents
        """
        all_chunks = []
        
        for doc_name, text in documents.items():
            doc_chunks = self.chunk_text(text, source=doc_name)
            all_chunks.extend(doc_chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def estimate_token_count(self, text: str) -> int:
        """
        Rough estimate of token count (1 token ≈ 4 characters for English)
        """
        return len(text) // 4
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict:
        """
        Get statistics about the chunks
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        token_counts = [chunk.token_count or 0 for chunk in chunks]
        
        sources = list(set(chunk.source for chunk in chunks))
        
        stats = {
            'total_chunks': len(chunks),
            'total_sources': len(sources),
            'sources': sources,
            'avg_chunk_length': sum(chunk_lengths) / len(chunks),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'total_characters': sum(chunk_lengths),
            'estimated_total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks) if token_counts else 0,
        }
        
        return stats
    
    def filter_chunks_by_length(self, 
                               chunks: List[TextChunk], 
                               min_length: int = 50, 
                               max_length: Optional[int] = None) -> List[TextChunk]:
        """
        Filter chunks by length to remove very short or very long chunks
        """
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_length = len(chunk.content)
            
            if chunk_length < min_length:
                continue
                
            if max_length and chunk_length > max_length:
                continue
                
            filtered_chunks.append(chunk)
        
        removed_count = len(chunks) - len(filtered_chunks)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} chunks based on length criteria")
        
        return filtered_chunks


class AdvancedTextChunker(TextChunker):
    """
    Advanced chunker with additional features
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def chunk_with_context(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Enhanced chunking that tries to preserve context
        """
        # First, try to split by major sections (double newlines)
        sections = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        start_pos = 0
        
        for section in sections:
            # If adding this section would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk + section) > self.chunk_size:
                if current_chunk.strip():
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        start_index=start_pos,
                        end_index=start_pos + len(current_chunk),
                        token_count=self.estimate_token_count(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    start_pos += len(current_chunk)
                
                # Start new chunk with overlap if needed
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + section
                else:
                    current_chunk = section
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                start_index=start_pos,
                end_index=start_pos + len(current_chunk),
                token_count=self.estimate_token_count(current_chunk)
            )
            chunks.append(chunk)
        
        # If no sections found or chunks are still too large, fall back to regular chunking
        if not chunks or any(len(chunk.content) > self.chunk_size * 1.5 for chunk in chunks):
            logger.info(f"Falling back to regular chunking for {source}")
            return super().chunk_text(text, source)
        
        logger.info(f"Context-aware chunking created {len(chunks)} chunks for {source}")
        return chunks


if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    Machine Learning Introduction
    
    Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
    
    Types of Machine Learning
    
    There are several types of machine learning algorithms, each with unique strengths and weaknesses:
    
    Supervised Learning: This type uses labeled training data to learn a mapping function from input variables to output variables.
    
    Unsupervised Learning: This finds hidden patterns or intrinsic structures in input data without labeled examples.
    
    Reinforcement Learning: An agent learns to behave in an environment by performing actions and seeing the results.
    """
    
    # Test regular chunker
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text, "sample.txt")
    
    print("=== Regular Chunking ===")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk.content)} chars")
        print(f"Content: {chunk.content[:100]}...")
        print()
    
    # Test advanced chunker
    advanced_chunker = AdvancedTextChunker(chunk_size=200, chunk_overlap=50)
    advanced_chunks = advanced_chunker.chunk_with_context(sample_text, "sample.txt")
    
    print("=== Advanced Chunking ===")
    for i, chunk in enumerate(advanced_chunks):
        print(f"Chunk {i}: {len(chunk.content)} chars")
        print(f"Content: {chunk.content[:100]}...")
        print()
    
    # Statistics
    stats = chunker.get_chunk_statistics(chunks)
    print("=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
