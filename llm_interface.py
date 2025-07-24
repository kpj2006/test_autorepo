"""
LLM Interface Module for RAG Assistant
Handles interactions with open-source LLMs via Groq and HuggingFace APIs
"""
import logging
import requests
import json
from typing import List, Dict, Optional, Tuple
from text_chunker import TextChunk
from groq import Groq
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    Template for RAG prompts
    """
    
    @staticmethod
    def create_rag_prompt(query: str, context_chunks: List[TextChunk], 
                         max_context_length: int = 2000) -> str:
        """
        Create a RAG prompt with context and query
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            max_context_length: Maximum length of context to include
            
        Returns:
            Formatted prompt string
        """
        # Build context from chunks
        context_parts = []
        current_length = 0
        
        for chunk in context_chunks:
            chunk_text = f"Source: {chunk.source}\nContent: {chunk.content}\n"
            if current_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n---\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. If the context doesn't contain sufficient information, clearly state this
3. Be specific and cite relevant sources when possible
4. If you need to use general knowledge beyond the context, clearly indicate this
5. Provide a clear, well-structured response

ANSWER:"""

        return prompt
    
    @staticmethod
    def create_no_context_prompt(query: str) -> str:
        """
        Create a prompt when no relevant context is found
        """
        return f"""You are a helpful AI assistant. The user has asked a question, but no relevant context was found in the available documents.

QUESTION: {query}

INSTRUCTIONS:
1. Acknowledge that you don't have specific information about this topic in the provided documents
2. If appropriate, provide general knowledge about the topic
3. Suggest that the user might want to provide more specific documents or rephrase their question
4. Be helpful while being clear about the limitations

ANSWER:"""


class GroqLLMClient:
    """
    Client for Groq API
    """
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        if not api_key:
            raise ValueError("Groq API key is required")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Groq client with model: {model}")
    
    def generate_response(self, prompt: str, 
                         max_tokens: int = 1000, 
                         temperature: float = 0.1,
                         timeout: int = 30) -> str:
        """
        Generate response using Groq API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response with Groq ({self.model})")
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"Successfully generated response ({len(generated_text)} characters)")
            return generated_text
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class HuggingFaceLLMClient:
    """
    Client for HuggingFace Inference API
    """
    
    def __init__(self, api_key: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize HuggingFace client
        
        Args:
            api_key: HuggingFace API key
            model: Model name to use
        """
        if not api_key:
            raise ValueError("HuggingFace API key is required")
        
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(f"Initialized HuggingFace client with model: {model}")
    
    def generate_response(self, prompt: str, 
                         max_tokens: int = 1000, 
                         temperature: float = 0.1,
                         timeout: int = 30,
                         retry_attempts: int = 3) -> str:
        """
        Generate response using HuggingFace Inference API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            
        Returns:
            Generated response text
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        for attempt in range(retry_attempts):
            try:
                logger.info(f"Generating response with HuggingFace ({self.model}) - Attempt {attempt + 1}")
                
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    logger.warning("Model is loading, waiting before retry...")
                    time.sleep(20)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "").strip()
                    logger.info(f"Successfully generated response ({len(generated_text)} characters)")
                    return generated_text
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out on attempt {attempt + 1}")
                if attempt == retry_attempts - 1:
                    raise
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"HuggingFace API error on attempt {attempt + 1}: {e}")
                if attempt == retry_attempts - 1:
                    raise
                time.sleep(2)
        
        raise Exception("All retry attempts failed")


class LLMInterface:
    """
    Unified interface for different LLM providers
    """
    
    def __init__(self, provider: str = "groq", 
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        """
        Initialize LLM interface
        
        Args:
            provider: LLM provider ("groq" or "huggingface")
            api_key: API key for the provider
            model: Model name to use
        """
        self.provider = provider.lower()
        
        if not api_key:
            raise ValueError(f"API key is required for {provider}")
        
        if self.provider == "groq":
            default_model = "mixtral-8x7b-32768"
            self.client = GroqLLMClient(api_key, model or default_model)
        elif self.provider == "huggingface":
            default_model = "mistralai/Mistral-7B-Instruct-v0.1"
            self.client = HuggingFaceLLMClient(api_key, model or default_model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM interface with {provider}")
    
    def generate_answer(self, query: str, context_chunks: List[TextChunk],
                       max_tokens: int = 1000,
                       temperature: float = 0.1,
                       similarity_threshold: float = 0.3) -> Dict[str, any]:
        """
        Generate an answer using RAG
        
        Args:
            query: User's question
            context_chunks: List of relevant chunks with similarity scores
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            similarity_threshold: Minimum similarity score to use context
            
        Returns:
            Dictionary with answer and metadata
        """
        # Filter chunks by similarity threshold
        relevant_chunks = [chunk for chunk, score in context_chunks if score >= similarity_threshold]
        
        if relevant_chunks:
            # Create RAG prompt with context
            prompt = PromptTemplate.create_rag_prompt(query, relevant_chunks)
            context_used = True
            logger.info(f"Using {len(relevant_chunks)} relevant chunks for context")
        else:
            # No relevant context found
            prompt = PromptTemplate.create_no_context_prompt(query)
            context_used = False
            logger.info("No relevant context found, using general prompt")
        
        # Generate response
        try:
            response = self.client.generate_response(
                prompt, 
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            # Extract sources from context chunks
            sources = list(set(chunk.source for chunk in relevant_chunks)) if relevant_chunks else []
            
            result = {
                "answer": response,
                "context_used": context_used,
                "sources": sources,
                "num_chunks_used": len(relevant_chunks),
                "query": query,
                "provider": self.provider,
                "model": self.client.model
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while generating the response: {str(e)}",
                "context_used": False,
                "sources": [],
                "num_chunks_used": 0,
                "query": query,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test the connection to the LLM provider
        
        Returns:
            True if connection is successful
        """
        try:
            test_response = self.client.generate_response(
                "Hello, this is a test. Please respond with 'Connection successful.'",
                max_tokens=50,
                temperature=0.1
            )
            logger.info(f"Connection test successful: {test_response[:100]}...")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Web search fallback (optional)
class WebSearchFallback:
    """
    Optional web search fallback using SerpAPI
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search fallback
        
        Args:
            api_key: SerpAPI key for web search
        """
        self.api_key = api_key
        self.enabled = bool(api_key)
        
        if self.enabled:
            logger.info("Web search fallback enabled")
        else:
            logger.info("Web search fallback disabled (no API key)")
    
    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for additional context
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with title, snippet, and link
        """
        if not self.enabled:
            return []
        
        try:
            from serpapi import GoogleSearch
            
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "engine": "google"
            })
            
            results = search.get_dict()
            organic_results = results.get("organic_results", [])
            
            formatted_results = []
            for result in organic_results[:num_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "link": result.get("link", "")
                })
            
            logger.info(f"Retrieved {len(formatted_results)} web search results")
            return formatted_results
            
        except ImportError:
            logger.warning("SerpAPI not installed. Install with: pip install google-search-results")
            return []
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []


if __name__ == "__main__":
    # Test the LLM interface (requires API keys)
    from config import GROQ_API_KEY, HUGGINGFACE_API_KEY
    
    if GROQ_API_KEY:
        print("=== Testing Groq LLM ===")
        try:
            groq_llm = LLMInterface("groq", GROQ_API_KEY)
            if groq_llm.test_connection():
                # Test sample chunks
                sample_chunks = [
                    (TextChunk("Machine learning is a subset of AI", "test.txt", 0, 0, 50, 12), 0.8),
                    (TextChunk("Python is used for data science", "test.txt", 1, 51, 100, 8), 0.6)
                ]
                
                result = groq_llm.generate_answer("What is machine learning?", sample_chunks)
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Groq test failed: {e}")
    
    if HUGGINGFACE_API_KEY:
        print("\n=== Testing HuggingFace LLM ===")
        try:
            hf_llm = LLMInterface("huggingface", HUGGINGFACE_API_KEY)
            if hf_llm.test_connection():
                result = hf_llm.generate_answer("What is Python?", [])
                print(f"Answer: {result['answer'][:200]}...")
        except Exception as e:
            print(f"HuggingFace test failed: {e}")
