"""
Example script demonstrating the RAG Assistant usage
"""
import os
from pathlib import Path
from rag_assistant import RAGAssistant

def main():
    """
    Example usage of the RAG Assistant
    """
    print("🤖 RAG Assistant Example")
    print("=" * 50)
    
    # Check if API keys are available
    groq_key = os.getenv("GROQ_API_KEY")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not groq_key and not hf_key:
        print("❌ No API keys found!")
        print("Please set GROQ_API_KEY or HUGGINGFACE_API_KEY environment variable")
        print()
        print("For Groq API:")
        print("1. Sign up at https://console.groq.com/")
        print("2. Create an API key")
        print("3. Set: $env:GROQ_API_KEY='your_api_key_here'")
        return
    
    # Configuration override for this example
    config_override = {
        'llm_provider': 'groq' if groq_key else 'huggingface',
        'groq_api_key': groq_key,
        'huggingface_api_key': hf_key,
    }
    
    # Initialize the assistant
    print("\n📦 Initializing RAG Assistant...")
    assistant = RAGAssistant(config_override)
    
    if not assistant.initialize():
        print("❌ Failed to initialize assistant")
        return
    
    print("✅ Assistant initialized successfully!")
    
    # Ingest documents (this will create sample documents if none exist)
    print("\n📚 Processing documents...")
    if not assistant.ingest_documents():
        print("❌ Failed to process documents")
        return
    
    print("✅ Documents processed successfully!")
    
    # Example queries
    example_questions = [
        "What is machine learning?",
        "What are the main types of machine learning?",
        "How is Python used in data science?",
        "What is the data science process?",
    ]
    
    print("\n" + "=" * 50)
    print("🔍 Example Queries")
    print("=" * 50)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 60)
        
        try:
            result = assistant.query(question)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display the answer
            print(f"🤖 Answer:")
            print(result.get('answer', 'No answer generated'))
            
            # Display metadata
            if result.get('context_used'):
                print(f"\n📄 Used {result.get('num_chunks_used', 0)} relevant chunks")
                print(f"📚 Sources: {', '.join(result.get('sources', []))}")
            else:
                print("\nℹ️ No relevant context found - used general knowledge")
            
            print(f"⏱️ Processing time: {result.get('processing_time', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
        
        print()
    
    print("=" * 50)
    print("💡 Try the interactive mode for a better experience:")
    print("   python cli.py interactive")
    print()
    print("Or ask specific questions:")
    print("   python cli.py query 'Your question here'")
    print("=" * 50)


if __name__ == "__main__":
    main()
