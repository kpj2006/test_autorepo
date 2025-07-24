"""
Main RAG Assistant Class
Orchestrates PDF processing, text chunking, vector storage, and LLM interaction
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from config import *
from pdf_processor import PDFProcessor, create_sample_pdfs
from text_chunker import TextChunker, AdvancedTextChunker, TextChunk
from vector_store import VectorStore
from llm_interface import LLMInterface, WebSearchFallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAssistant:
    """
    Main RAG Assistant class that coordinates all components
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize the RAG Assistant
        
        Args:
            config_override: Optional configuration overrides
        """
        self.console = Console()
        
        # Load configuration
        self.config = self._load_config(config_override)
        
        # Initialize components
        self.pdf_processor = None
        self.text_chunker = None
        self.vector_store = None
        self.llm_interface = None
        self.web_search = None
        
        # State
        self.documents = {}
        self.chunks = []
        self.initialized = False
        
        self.console.print("[bold green]RAG Assistant initialized[/bold green]")
    
    def _load_config(self, config_override: Optional[Dict] = None) -> Dict:
        """Load and validate configuration"""
        config = {
            'docs_dir': DOCS_DIR,
            'vector_db_dir': VECTOR_DB_DIR,
            'embeddings_model_cache': EMBEDDINGS_MODEL_CACHE,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'embedding_model_name': EMBEDDING_MODEL_NAME,
            'vector_db_type': VECTOR_DB_TYPE,
            'top_k_results': TOP_K_RESULTS,
            'llm_provider': LLM_PROVIDER,
            'groq_model': GROQ_MODEL,
            'hf_model': HF_MODEL,
            'groq_api_key': GROQ_API_KEY,
            'huggingface_api_key': HUGGINGFACE_API_KEY,
            'serpapi_key': SERPAPI_KEY,
            'max_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
            'web_search_threshold': WEB_SEARCH_THRESHOLD,
            'enable_web_search': ENABLE_WEB_SEARCH,
        }
        
        if config_override:
            config.update(config_override)
        
        return config
    
    def initialize(self) -> bool:
        """
        Initialize all components of the RAG system
        
        Returns:
            True if initialization successful
        """
        try:
            self.console.print("\n[bold blue]Initializing RAG Assistant Components...[/bold blue]")
            
            # Initialize PDF processor
            self.console.print("📄 Initializing PDF processor...")
            self.pdf_processor = PDFProcessor(self.config['docs_dir'])
            
            # Initialize text chunker
            self.console.print("✂️ Initializing text chunker...")
            self.text_chunker = AdvancedTextChunker(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            
            # Initialize vector store
            self.console.print("🔍 Initializing vector store...")
            self.vector_store = VectorStore(
                store_type=self.config['vector_db_type'],
                embedding_model=self.config['embedding_model_name'],
                persist_directory=self.config['vector_db_dir'],
                model_cache_dir=self.config['embeddings_model_cache']
            )
            
            # Initialize LLM interface
            self.console.print("🤖 Initializing LLM interface...")
            api_key = None
            model = None
            
            if self.config['llm_provider'].lower() == 'groq':
                api_key = self.config['groq_api_key']
                model = self.config['groq_model']
            elif self.config['llm_provider'].lower() == 'huggingface':
                api_key = self.config['huggingface_api_key']
                model = self.config['hf_model']
            
            if not api_key:
                raise ValueError(f"No API key found for {self.config['llm_provider']}")
            
            self.llm_interface = LLMInterface(
                provider=self.config['llm_provider'],
                api_key=api_key,
                model=model
            )
            
            # Initialize web search fallback
            if self.config['enable_web_search'] and self.config['serpapi_key']:
                self.console.print("🌐 Initializing web search fallback...")
                self.web_search = WebSearchFallback(self.config['serpapi_key'])
            
            # Test LLM connection
            self.console.print("🔗 Testing LLM connection...")
            if not self.llm_interface.test_connection():
                self.console.print("[red]Warning: LLM connection test failed[/red]")
            
            self.initialized = True
            self.console.print("[bold green]✅ All components initialized successfully![/bold green]")
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Initialization failed: {e}[/bold red]")
            logger.error(f"Initialization error: {e}")
            return False
    
    def ingest_documents(self, force_reprocess: bool = False) -> bool:
        """
        Ingest documents from the docs directory
        
        Args:
            force_reprocess: Force reprocessing even if vector DB exists
            
        Returns:
            True if ingestion successful
        """
        if not self.initialized:
            self.console.print("[red]System not initialized. Call initialize() first.[/red]")
            return False
        
        try:
            self.console.print("\n[bold blue]📚 Starting document ingestion...[/bold blue]")
            
            # Check if we can load existing vector store
            if self.vector_store is not None and not force_reprocess and self.vector_store.load():
                self.console.print("[green]✅ Loaded existing vector database[/green]")
                return True
            
            # Create sample documents if no PDFs exist
            create_sample_pdfs(self.config['docs_dir'])
            
            # Process PDFs
            self.console.print("📄 Processing documents...")
            if self.pdf_processor is None:
                self.pdf_processor = PDFProcessor(self.config['docs_dir'])
            self.documents = self.pdf_processor.process_all_pdfs()
            
            # Also process any text files for testing
            text_files = list(self.config['docs_dir'].glob("*.txt"))
            for text_file in text_files:
                try:
                    content = text_file.read_text(encoding='utf-8')
                    self.documents[text_file.name] = content
                    logger.info(f"Processed text file: {text_file.name}")
                except Exception as e:
                    logger.error(f"Error processing {text_file.name}: {e}")
            
            if not self.documents:
                self.console.print("[red]❌ No documents found to process[/red]")
                return False
            
            # Display document statistics
            self._display_document_stats()
            
            # Chunk documents
            self.console.print("✂️ Chunking documents...")
            if self.text_chunker is None:
                self.text_chunker = AdvancedTextChunker(
                    chunk_size=self.config['chunk_size'],
                    chunk_overlap=self.config['chunk_overlap']
                )
            self.chunks = self.text_chunker.chunk_documents(self.documents)
            
            # Display chunk statistics
            self._display_chunk_stats()
            
            # Add to vector store
            self.console.print("🔍 Creating embeddings and storing in vector database...")
            if self.vector_store is None:
                self.vector_store = VectorStore(
                    store_type=self.config['vector_db_type'],
                    embedding_model=self.config['embedding_model_name'],
                    persist_directory=self.config['vector_db_dir'],
                    model_cache_dir=self.config['embeddings_model_cache']
                )
            self.vector_store.add_chunks(self.chunks)
            
            # Save vector store
            self.console.print("💾 Saving vector database...")
            self.vector_store.save()
            
            self.console.print("[bold green]✅ Document ingestion completed successfully![/bold green]")
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]❌ Document ingestion failed: {e}[/bold red]")
            logger.error(f"Ingestion error: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.initialized:
            return {"error": "System not initialized"}
        
        if not self.chunks:
            return {"error": "No documents ingested"}
        
        try:
            start_time = time.time()
            
            # Search for relevant chunks
            self.console.print(f"🔍 Searching for relevant information...")
            search_results = self.vector_store.search(question, k=self.config['top_k_results'])
            
            if not search_results:
                self.console.print("[yellow]⚠️ No relevant chunks found[/yellow]")
                return self.llm_interface.generate_answer(question, [])
            
            # Get the best similarity score
            max_score = max(score for _, score in search_results)
            
            # Use web search fallback if similarity is too low
            if (self.web_search and 
                max_score < self.config['web_search_threshold']):
                self.console.print(f"🌐 Similarity too low ({max_score:.3f}), searching web...")
                web_results = self.web_search.search_web(question, 3)
                if web_results:
                    self.console.print(f"Found {len(web_results)} web results")
            
            # Generate answer
            self.console.print(f"🤖 Generating answer...")
            result = self.llm_interface.generate_answer(
                question, 
                search_results,
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            
            # Add timing and search info
            result['processing_time'] = time.time() - start_time
            result['search_results'] = [
                {
                    'content': chunk.content[:200] + "...",
                    'source': chunk.source,
                    'similarity': score
                }
                for chunk, score in search_results
            ]
            
            return result
            
        except Exception as e:
            error_msg = f"Query failed: {e}"
            self.console.print(f"[bold red]❌ {error_msg}[/bold red]")
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _display_document_stats(self):
        """Display document processing statistics"""
        table = Table(title="Document Statistics")
        table.add_column("Document", style="cyan")
        table.add_column("Characters", style="green")
        table.add_column("Estimated Tokens", style="yellow")
        
        total_chars = 0
        for doc_name, content in self.documents.items():
            char_count = len(content)
            token_count = char_count // 4  # Rough estimate
            total_chars += char_count
            
            table.add_row(
                doc_name,
                f"{char_count:,}",
                f"{token_count:,}"
            )
        
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_chars:,}[/bold]",
            f"[bold]{total_chars // 4:,}[/bold]"
        )
        
        self.console.print(table)
    
    def _display_chunk_stats(self):
        """Display chunk statistics"""
        if not self.chunks:
            return
        
        stats = self.text_chunker.get_chunk_statistics(self.chunks)
        
        info_panel = Panel.fit(
            f"""[bold]Chunk Statistics[/bold]
            
📊 Total Chunks: {stats['total_chunks']}
📄 Sources: {stats['total_sources']}
📏 Avg Length: {stats['avg_chunk_length']:.0f} chars
📉 Min Length: {stats['min_chunk_length']} chars
📈 Max Length: {stats['max_chunk_length']} chars
🎯 Total Characters: {stats['total_characters']:,}
🔢 Estimated Tokens: {stats['estimated_total_tokens']:,}""",
            title="Processing Results",
            border_style="blue"
        )
        
        self.console.print(info_panel)
    
    def display_system_info(self):
        """Display system information"""
        info = {
            "Configuration": {
                "Vector DB Type": self.config['vector_db_type'],
                "Embedding Model": self.config['embedding_model_name'],
                "LLM Provider": self.config['llm_provider'],
                "Chunk Size": self.config['chunk_size'],
                "Chunk Overlap": self.config['chunk_overlap'],
                "Top K Results": self.config['top_k_results']
            },
            "Status": {
                "Initialized": "✅" if self.initialized else "❌",
                "Documents Loaded": len(self.documents),
                "Total Chunks": len(self.chunks),
                "Web Search": "✅" if self.web_search and self.web_search.enabled else "❌"
            }
        }
        
        for section, items in info.items():
            table = Table(title=section)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in items.items():
                table.add_row(key, str(value))
            
            self.console.print(table)
    
    def interactive_mode(self):
        """
        Run the assistant in interactive mode
        """
        if not self.initialized:
            self.console.print("[red]Please initialize the system first[/red]")
            return
        
        self.console.print("\n[bold green]🤖 RAG Assistant Interactive Mode[/bold green]")
        self.console.print("Type your questions below. Commands:")
        self.console.print("• 'quit' or 'exit' to quit")
        self.console.print("• 'info' to show system information")
        self.console.print("• 'reload' to reload documents")
        self.console.print()
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[yellow]Goodbye! 👋[/yellow]")
                    break
                
                if question.lower() == 'info':
                    self.display_system_info()
                    continue
                
                if question.lower() == 'reload':
                    self.console.print("\n🔄 Reloading documents...")
                    if self.ingest_documents(force_reprocess=True):
                        self.console.print("[green]✅ Documents reloaded successfully[/green]")
                    continue
                
                if not question:
                    continue
                
                # Process query
                with self.console.status("[bold green]Processing your question...", spinner="dots"):
                    result = self.query(question)
                
                # Display results
                self._display_query_result(result)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def _display_query_result(self, result: Dict[str, Any]):
        """Display query results in a formatted way"""
        if "error" in result:
            self.console.print(f"[red]❌ {result['error']}[/red]")
            return
        
        # Main answer
        answer_panel = Panel(
            result.get('answer', 'No answer generated'),
            title=f"🤖 Answer ({result.get('provider', 'unknown')})",
            border_style="green" if result.get('context_used') else "yellow"
        )
        self.console.print(answer_panel)
        
        # Metadata
        metadata_text = []
        if result.get('context_used'):
            metadata_text.append(f"📄 Used {result.get('num_chunks_used', 0)} relevant chunks")
            metadata_text.append(f"📚 Sources: {', '.join(result.get('sources', []))}")
        else:
            metadata_text.append("ℹ️ No relevant context found - using general knowledge")
        
        if 'processing_time' in result:
            metadata_text.append(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        
        if metadata_text:
            self.console.print(Panel(
                "\n".join(metadata_text),
                title="Query Details",
                border_style="blue"
            ))


def main():
    """Main function for testing"""
    assistant = RAGAssistant()
    
    if assistant.initialize():
        if assistant.ingest_documents():
            assistant.interactive_mode()
        else:
            print("Failed to ingest documents")
    else:
        print("Failed to initialize assistant")


if __name__ == "__main__":
    main()
