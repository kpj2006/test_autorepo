# 🤖 RAG Assistant - Retrieval-Augmented Generation Q&A System

A powerful, open-source **Retrieval-Augmented Generation (RAG)** question-answering assistant that runs entirely with open-source tools and Python. No frontend interface required - works completely through the command line!

## 🌟 Features

- **📄 PDF Processing**: Extracts text from PDF files using PyMuPDF and pdfplumber with fallback support
- **✂️ Smart Chunking**: Uses LangChain's RecursiveCharacterTextSplitter with overlapping chunks for context preservation
- **🔍 Vector Search**: Supports both FAISS and ChromaDB for similarity-based document retrieval
- **🧠 Open-Source LLMs**: Integrates with Groq API and HuggingFace Inference API (Mistral, LLaMA, etc.)
- **🌐 Web Search Fallback**: Optional SerpAPI integration when document context is insufficient
- **💬 Interactive CLI**: Beautiful command-line interface with rich formatting
- **⚡ Fast & Local**: All processing happens locally with cached models
- **🛠️ Configurable**: Easily customizable parameters and model choices

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone or create the project directory
cd rag_assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Get API Keys

You need at least one LLM API key:

**Option A - Groq (Recommended - Fast & Free tier available)**
1. Sign up at [Groq Console](https://console.groq.com/)
2. Create an API key
3. Set environment variable: `$env:GROQ_API_KEY="your_api_key_here"`

**Option B - HuggingFace**
1. Sign up at [HuggingFace](https://huggingface.co/)
2. Create an access token
3. Set environment variable: `$env:HUGGINGFACE_API_KEY="your_token_here"`

### 3. Add Your Documents

Place your PDF files in the `./docs/` directory:

```powershell
# Example: copy your PDFs
Copy-Item "C:\path\to\your\document.pdf" ".\docs\"
```

### 4. Initialize the System

```powershell
# Quick setup with default settings
python cli.py setup

# Or with custom options
python cli.py setup --llm-provider groq --vector-db FAISS
```

### 5. Start Asking Questions!

```powershell
# Interactive mode (recommended)
python cli.py interactive

# Or ask a single question
python cli.py query "What is the main topic discussed in the documents?"
```

## 📋 Detailed Setup Guide

### Prerequisites

- **Python 3.8+**
- **Windows PowerShell** (or adapt commands for your shell)
- **At least 4GB RAM** (for local embeddings)
- **Internet connection** (for downloading models and API calls)

### Step-by-Step Installation

1. **Create project directory:**
   ```powershell
   mkdir rag_assistant
   cd rag_assistant
   ```

2. **Install Python dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```powershell
   # Create environment file template
   python cli.py config --create-env-file
   
   # Edit the .env.template file and save as .env
   # Or set variables directly:
   $env:GROQ_API_KEY="your_groq_api_key"
   ```

4. **Test the installation:**
   ```powershell
   python cli.py test
   ```

5. **View system information:**
   ```powershell
   python cli.py info
   ```

## 🎯 Usage Examples

### Command Line Interface

```powershell
# Show all available commands
python cli.py --help

# Setup the system (first time)
python cli.py setup --llm-provider groq

# Process documents
python cli.py ingest --docs-dir ./my-documents

# Ask questions
python cli.py query "What are the key findings?"
python cli.py query "Summarize the main conclusions" --json-output

# Interactive chat
python cli.py interactive

# System information
python cli.py info
python cli.py test
```

### Interactive Mode Commands

When in interactive mode, you can use these special commands:

- `quit` or `exit` - Exit the program
- `info` - Show system information
- `reload` - Reload and reprocess documents

### Python API Usage

```python
from rag_assistant import RAGAssistant

# Initialize
assistant = RAGAssistant()
assistant.initialize()
assistant.ingest_documents()

# Ask questions
result = assistant.query("What is machine learning?")
print(result['answer'])
```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes* | Groq API key for LLM access |
| `HUGGINGFACE_API_KEY` | Yes* | HuggingFace API token |
| `SERPAPI_KEY` | No | SerpAPI key for web search fallback |

*At least one LLM API key is required

### Configuration Options

Edit `config.py` to customize:

```python
# Text Processing
CHUNK_SIZE = 1000          # Size of text chunks
CHUNK_OVERLAP = 200        # Overlap between chunks

# Vector Database
VECTOR_DB_TYPE = "FAISS"   # Options: "FAISS", "ChromaDB"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Settings
LLM_PROVIDER = "groq"      # Options: "groq", "huggingface"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 1000
TEMPERATURE = 0.1

# Retrieval
TOP_K_RESULTS = 5          # Number of chunks to retrieve
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│  Text Chunking  │───▶│  Embeddings     │
│   (.pdf, .txt)  │    │  (LangChain)    │    │ (SentenceT5)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Vector Search   │◀───│  Vector Store   │
│                 │    │ (Similarity)    │    │ (FAISS/Chroma)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │
          │              ┌─────────────────┐
          │              │ Retrieved Chunks│
          │              │   + Context     │
          │              └─────────────────┘
          │                       │
          ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Interface                                │
│            (Groq API / HuggingFace API)                        │
│        Mixtral-8x7B / Mistral-7B / LLaMA-2                    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐
│  Final Answer   │
│   + Sources     │
└─────────────────┘
```

## 🛠️ Available Models

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `BAAI/bge-small-en` (alternative, good quality)

### LLM Models

**Groq (Recommended):**
- `mixtral-8x7b-32768` (default, excellent performance)
- `llama2-70b-4096` (high quality, slower)

**HuggingFace:**
- `mistralai/Mistral-7B-Instruct-v0.1`
- `microsoft/DialoGPT-medium`
- Any compatible model from HuggingFace Hub

## 🔧 Troubleshooting

### Common Issues

**1. "No API key found"**
```powershell
# Set your API key
$env:GROQ_API_KEY="your_api_key_here"

# Or create .env file
python cli.py config --create-env-file
```

**2. "Failed to load model"**
- Check internet connection
- Verify disk space (models can be several GB)
- Try a different embedding model

**3. "No documents found"**
- Ensure PDF files are in the `./docs/` directory
- Check file permissions
- Try with sample documents: `python cli.py setup`

**4. "Connection failed"**
- Verify API keys are correct
- Check internet connectivity
- Try different LLM provider

### Performance Optimization

**For faster processing:**
- Use FAISS instead of ChromaDB
- Reduce chunk size and overlap
- Use smaller embedding models

**For better quality:**
- Increase chunk overlap
- Use larger embedding models
- Increase TOP_K_RESULTS

## 📁 Project Structure

```
rag_assistant/
├── config.py              # Configuration settings
├── pdf_processor.py       # PDF text extraction
├── text_chunker.py        # Text chunking logic
├── vector_store.py        # Vector database operations
├── llm_interface.py       # LLM API integration
├── rag_assistant.py       # Main RAG orchestration
├── cli.py                # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── docs/                 # Your PDF documents
├── vector_db/            # Persistent vector database
├── models/               # Cached embedding models
└── .env                  # Environment variables
```

## 🚀 Advanced Features

### Web Search Fallback

Enable web search when document context is insufficient:

```python
# Set SerpAPI key
$env:SERPAPI_KEY="your_serpapi_key"

# System will automatically fall back to web search
# when similarity scores are below threshold
```

### Custom Document Processing

```python
from pdf_processor import PDFProcessor

processor = PDFProcessor("./custom-docs")
texts = processor.process_all_pdfs()
```

### Batch Processing

```powershell
# Process multiple document sets
python cli.py ingest --docs-dir ./set1
python cli.py ingest --docs-dir ./set2
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♀️ Support

- **Issues**: Create a GitHub issue
- **Questions**: Check existing issues or create a new one
- **Documentation**: This README and inline code comments

---

**Happy questioning! 🤖✨**
