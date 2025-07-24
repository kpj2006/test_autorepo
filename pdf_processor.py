"""
PDF Processing Module for RAG Assistant
Handles PDF text extraction using multiple libraries for robustness
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import pdfplumber
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF text extraction with fallback methods
    """
    
    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
        
    def extract_text_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (faster, good for most PDFs)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"pdfplumber failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF with fallback methods
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Try PyMuPDF first (faster)
        text = self.extract_text_pymupdf(pdf_path)
        
        # If PyMuPDF fails or returns empty, try pdfplumber
        if not text or len(text.strip()) < 100:
            logger.info(f"Trying pdfplumber for {pdf_path.name}")
            text = self.extract_text_pdfplumber(pdf_path)
        
        if not text:
            logger.warning(f"Failed to extract text from {pdf_path.name}")
            
        return text
    
    def process_all_pdfs(self) -> Dict[str, str]:
        """
        Process all PDF files in the docs directory
        Returns a dictionary mapping filename to extracted text
        """
        pdf_texts = {}
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.docs_dir}")
            return pdf_texts
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    pdf_texts[pdf_path.name] = text
                    logger.info(f"Successfully processed {pdf_path.name} ({len(text)} characters)")
                else:
                    logger.warning(f"No text extracted from {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
        
        return pdf_texts
    
    def get_document_metadata(self, pdf_path: Path) -> Dict[str, any]:
        """Extract metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            doc.close()
            
            return {
                'filename': pdf_path.name,
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'file_size': pdf_path.stat().st_size,
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {'filename': pdf_path.name}


def create_sample_pdfs(docs_dir: Path):
    """
    Create some sample text files for testing if no PDFs are present
    """
    if not any(docs_dir.glob("*.pdf")):
        logger.info("No PDFs found. Creating sample text files for testing...")
        
        sample_texts = {
            "machine_learning.txt": """
            Machine Learning Fundamentals
            
            Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. The field encompasses various algorithms and techniques that allow systems to improve their performance on specific tasks through experience.
            
            There are three main types of machine learning:
            1. Supervised Learning: Uses labeled data to train models
            2. Unsupervised Learning: Finds patterns in unlabeled data  
            3. Reinforcement Learning: Learns through interaction with an environment
            
            Common algorithms include linear regression, decision trees, neural networks, and support vector machines.
            """,
            
            "python_programming.txt": """
            Python Programming Best Practices
            
            Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence.
            
            Key Python concepts include:
            - Variables and data types
            - Control structures (if/else, loops)
            - Functions and classes
            - Modules and packages
            - Exception handling
            
            Best practices include following PEP 8 style guidelines, writing docstrings, using virtual environments, and writing unit tests.
            """,
            
            "data_science.txt": """
            Introduction to Data Science
            
            Data science is an interdisciplinary field that combines statistics, mathematics, computer science, and domain expertise to extract insights from data.
            
            The data science process typically involves:
            1. Data collection and acquisition
            2. Data cleaning and preprocessing
            3. Exploratory data analysis
            4. Model building and validation
            5. Deployment and monitoring
            
            Popular tools include Python, R, SQL, Jupyter notebooks, and various libraries like pandas, numpy, scikit-learn, and matplotlib.
            """
        }
        
        for filename, content in sample_texts.items():
            file_path = docs_dir / filename
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"Created sample file: {filename}")


if __name__ == "__main__":
    from config import DOCS_DIR
    
    # Create sample files if no PDFs exist
    create_sample_pdfs(DOCS_DIR)
    
    # Test PDF processing
    processor = PDFProcessor(DOCS_DIR)
    texts = processor.process_all_pdfs()
    
    for filename, text in texts.items():
        print(f"\n=== {filename} ===")
        print(f"Length: {len(text)} characters")
        print(f"Preview: {text[:200]}...")
