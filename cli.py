"""
Command Line Interface for RAG Assistant
Provides easy-to-use CLI commands for different operations
"""
import click
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from config import *
from rag_assistant import RAGAssistant

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    🤖 RAG Assistant - Retrieval-Augmented Generation Q&A System
    
    A powerful document-based question-answering system using open-source tools.
    """
    pass


@cli.command()
@click.option('--docs-dir', default=str(DOCS_DIR), help='Directory containing PDF documents')
@click.option('--llm-provider', default=LLM_PROVIDER, type=click.Choice(['groq', 'huggingface']), 
              help='LLM provider to use')
@click.option('--vector-db', default=VECTOR_DB_TYPE, type=click.Choice(['FAISS', 'ChromaDB']),
              help='Vector database type')
@click.option('--force-reprocess', is_flag=True, help='Force reprocessing of documents')
@click.option('--groq-api-key', envvar='GROQ_API_KEY', help='Groq API key')
@click.option('--hf-api-key', envvar='HUGGINGFACE_API_KEY', help='HuggingFace API key')
def setup(docs_dir, llm_provider, vector_db, force_reprocess, groq_api_key, hf_api_key):
    """
    🚀 Initialize and setup the RAG Assistant
    """
    console.print("[bold blue]Setting up RAG Assistant...[/bold blue]")
    
    # Validate API key
    if llm_provider == 'groq' and not groq_api_key:
        console.print("[red]❌ Groq API key required. Set GROQ_API_KEY environment variable or use --groq-api-key[/red]")
        return
    
    if llm_provider == 'huggingface' and not hf_api_key:
        console.print("[red]❌ HuggingFace API key required. Set HUGGINGFACE_API_KEY environment variable or use --hf-api-key[/red]")
        return
    
    # Configuration override
    config_override = {
        'docs_dir': Path(docs_dir),
        'llm_provider': llm_provider,
        'vector_db_type': vector_db,
        'groq_api_key': groq_api_key,
        'huggingface_api_key': hf_api_key
    }
    
    # Initialize assistant
    assistant = RAGAssistant(config_override)
    
    if assistant.initialize():
        if assistant.ingest_documents(force_reprocess=force_reprocess):
            console.print("\n[bold green]✅ Setup completed successfully![/bold green]")
            console.print(f"📄 Documents directory: {docs_dir}")
            console.print(f"🤖 LLM Provider: {llm_provider}")
            console.print(f"🔍 Vector DB: {vector_db}")
        else:
            console.print("[red]❌ Failed to process documents[/red]")
    else:
        console.print("[red]❌ Failed to initialize system[/red]")


@cli.command()
@click.argument('question')
@click.option('--llm-provider', default=LLM_PROVIDER, type=click.Choice(['groq', 'huggingface']))
@click.option('--groq-api-key', envvar='GROQ_API_KEY', help='Groq API key')
@click.option('--hf-api-key', envvar='HUGGINGFACE_API_KEY', help='HuggingFace API key')
@click.option('--json-output', is_flag=True, help='Output result in JSON format')
def query(question, llm_provider, groq_api_key, hf_api_key, json_output):
    """
    ❓ Ask a question to the RAG Assistant
    
    QUESTION: The question you want to ask
    """
    # Validate API key
    if llm_provider == 'groq' and not groq_api_key:
        console.print("[red]❌ Groq API key required[/red]")
        return
    
    if llm_provider == 'huggingface' and not hf_api_key:
        console.print("[red]❌ HuggingFace API key required[/red]")
        return
    
    # Configuration
    config_override = {
        'llm_provider': llm_provider,
        'groq_api_key': groq_api_key,
        'huggingface_api_key': hf_api_key
    }
    
    # Initialize and run query
    assistant = RAGAssistant(config_override)
    
    if not assistant.initialize():
        console.print("[red]❌ Failed to initialize system[/red]")
        return
    
    # Try to load existing vector database
    if not assistant.vector_store.load():
        console.print("[yellow]⚠️ No existing knowledge base found. Run 'setup' command first.[/yellow]")
        return
    
    console.print(f"❓ Question: {question}")
    
    with console.status("[bold green]Processing question...", spinner="dots"):
        result = assistant.query(question)
    
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        assistant._display_query_result(result)


@cli.command()
def interactive():
    """
    💬 Start interactive chat mode
    """
    # Check for API keys
    if not GROQ_API_KEY and not HUGGINGFACE_API_KEY:
        console.print("[red]❌ No API keys found. Please set GROQ_API_KEY or HUGGINGFACE_API_KEY[/red]")
        return
    
    # Initialize assistant
    assistant = RAGAssistant()
    
    if assistant.initialize():
        # Try to load existing vector database
        if not assistant.vector_store.load():
            console.print("[yellow]⚠️ No existing knowledge base found.[/yellow]")
            if click.confirm("Would you like to process documents now?"):
                if not assistant.ingest_documents():
                    console.print("[red]❌ Failed to process documents[/red]")
                    return
            else:
                console.print("[red]Cannot continue without processed documents[/red]")
                return
        
        assistant.interactive_mode()
    else:
        console.print("[red]❌ Failed to initialize assistant[/red]")


@cli.command()
@click.option('--docs-dir', default=str(DOCS_DIR), help='Directory containing documents')
def ingest(docs_dir):
    """
    📚 Ingest and process documents
    """
    console.print("[bold blue]Processing documents...[/bold blue]")
    
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        console.print(f"[red]❌ Directory not found: {docs_dir}[/red]")
        return
    
    config_override = {'docs_dir': docs_path}
    assistant = RAGAssistant(config_override)
    
    if assistant.initialize():
        if assistant.ingest_documents(force_reprocess=True):
            console.print("[bold green]✅ Documents processed successfully![/bold green]")
        else:
            console.print("[red]❌ Failed to process documents[/red]")
    else:
        console.print("[red]❌ Failed to initialize system[/red]")


@cli.command()
def info():
    """
    ℹ️ Display system information and configuration
    """
    console.print("[bold blue]RAG Assistant System Information[/bold blue]\n")
    
    # Configuration table
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_items = [
        ("Documents Directory", str(DOCS_DIR)),
        ("Vector Database Type", VECTOR_DB_TYPE),
        ("Embedding Model", EMBEDDING_MODEL_NAME),
        ("LLM Provider", LLM_PROVIDER),
        ("Chunk Size", str(CHUNK_SIZE)),
        ("Chunk Overlap", str(CHUNK_OVERLAP)),
        ("Top K Results", str(TOP_K_RESULTS)),
        ("Max Tokens", str(MAX_TOKENS)),
        ("Temperature", str(TEMPERATURE)),
    ]
    
    for setting, value in config_items:
        config_table.add_row(setting, value)
    
    console.print(config_table)
    
    # API Keys status
    api_status = Table(title="API Keys Status")
    api_status.add_column("Provider", style="cyan")
    api_status.add_column("Status", style="green")
    
    api_status.add_row("Groq", "✅ Available" if GROQ_API_KEY else "❌ Not set")
    api_status.add_row("HuggingFace", "✅ Available" if HUGGINGFACE_API_KEY else "❌ Not set")
    api_status.add_row("SerpAPI (Web Search)", "✅ Available" if SERPAPI_KEY else "❌ Not set")
    
    console.print(api_status)
    
    # Directories status
    dirs_table = Table(title="Directories")
    dirs_table.add_column("Directory", style="cyan")
    dirs_table.add_column("Path", style="green")
    dirs_table.add_column("Exists", style="yellow")
    
    directories = [
        ("Documents", DOCS_DIR),
        ("Vector Database", VECTOR_DB_DIR),
        ("Model Cache", EMBEDDINGS_MODEL_CACHE),
    ]
    
    for name, path in directories:
        exists = "✅ Yes" if path.exists() else "❌ No"
        dirs_table.add_row(name, str(path), exists)
    
    console.print(dirs_table)


@cli.command()
@click.option('--create-env-file', is_flag=True, help='Create a .env template file')
def config(create_env_file):
    """
    ⚙️ Configuration management
    """
    if create_env_file:
        env_content = """# RAG Assistant Configuration
# Copy this file to .env and fill in your API keys

# LLM API Keys (at least one required)
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Optional: Web search fallback
SERPAPI_KEY=your_serpapi_key_here

# Optional: Custom paths
# DOCS_DIR=./docs
# VECTOR_DB_DIR=./vector_db
# MODEL_CACHE_DIR=./models
"""
        
        env_file = Path(".env.template")
        env_file.write_text(env_content)
        console.print(f"[green]✅ Created environment template: {env_file}[/green]")
        console.print("Copy this to '.env' and fill in your API keys")
    else:
        console.print("Use --create-env-file to create a configuration template")


@cli.command()
def test():
    """
    🧪 Test system functionality
    """
    console.print("[bold blue]Testing RAG Assistant...[/bold blue]")
    
    # Test 1: Check API keys
    console.print("\n1. Checking API keys...")
    if GROQ_API_KEY:
        console.print("  ✅ Groq API key found")
    if HUGGINGFACE_API_KEY:
        console.print("  ✅ HuggingFace API key found")
    if not GROQ_API_KEY and not HUGGINGFACE_API_KEY:
        console.print("  ❌ No API keys found")
        return
    
    # Test 2: Initialize system
    console.print("\n2. Testing system initialization...")
    assistant = RAGAssistant()
    
    if not assistant.initialize():
        console.print("  ❌ System initialization failed")
        return
    console.print("  ✅ System initialized successfully")
    
    # Test 3: Test LLM connection
    console.print("\n3. Testing LLM connection...")
    if assistant.llm_interface.test_connection():
        console.print("  ✅ LLM connection working")
    else:
        console.print("  ❌ LLM connection failed")
        return
    
    # Test 4: Create sample documents and ingest
    console.print("\n4. Testing document processing...")
    if assistant.ingest_documents():
        console.print("  ✅ Document processing successful")
    else:
        console.print("  ❌ Document processing failed")
        return
    
    # Test 5: Test query
    console.print("\n5. Testing query functionality...")
    test_question = "What is machine learning?"
    result = assistant.query(test_question)
    
    if "error" in result:
        console.print(f"  ❌ Query failed: {result['error']}")
        return
    
    console.print("  ✅ Query successful")
    console.print(f"  Answer preview: {result.get('answer', '')[:100]}...")
    
    console.print("\n[bold green]🎉 All tests passed! Your RAG Assistant is ready to use.[/bold green]")


if __name__ == "__main__":
    cli()
