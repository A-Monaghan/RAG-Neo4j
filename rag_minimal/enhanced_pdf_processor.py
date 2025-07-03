#!/usr/bin/env python3
"""
Enhanced PDF Processor for Neo4j RAG System

This module provides comprehensive PDF processing with:
1. PyMuPDF (fitz) for advanced text extraction
2. Layout-aware processing
3. Advanced text cleaning
4. Structured data preparation for Neo4j ingestion

Designed to work with your existing processed data and extend it.
"""

import fitz  # PyMuPDF
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure."""
    path: str
    filename: str
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = 0
    file_size: int = 0

@dataclass
class TextChunk:
    """Enhanced text chunk with metadata for Neo4j."""
    chunk_id: str
    text: str
    document_path: str
    chunk_index: int
    page_start: int
    page_end: int
    word_count: int
    char_count: int
    embedding: Optional[List[float]] = None
    confidence_score: float = 1.0

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, DocumentMetadata]:
    """
    Extract text from PDF using PyMuPDF (fitz) with enhanced metadata.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Tuple[str, DocumentMetadata]: Extracted text and document metadata
    """
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = DocumentMetadata(
            path=pdf_path,
            filename=Path(pdf_path).name,
            title=doc.metadata.get('title', ''),
            author=doc.metadata.get('author', ''),
            subject=doc.metadata.get('subject', ''),
            creator=doc.metadata.get('creator', ''),
            producer=doc.metadata.get('producer', ''),
            page_count=len(doc),
            file_size=Path(pdf_path).stat().st_size if Path(pdf_path).exists() else 0
        )
        
        # Handle dates safely
        try:
            if doc.metadata.get('creationDate'):
                metadata.creation_date = datetime.strptime(
                    doc.metadata['creationDate'].replace('D:', '').split('+')[0], 
                    '%Y%m%d%H%M%S'
                )
        except:
            pass
            
        try:
            if doc.metadata.get('modDate'):
                metadata.modification_date = datetime.strptime(
                    doc.metadata['modDate'].replace('D:', '').split('+')[0], 
                    '%Y%m%d%H%M%S'
                )
        except:
            pass
        
        # Extract text with page information
        full_text = ""
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Try different extraction methods
            try:
                # Method 1: Standard text extraction
                page_text = page.get_text()
                
                # Method 2: If standard fails, try text blocks
                if not page_text.strip():
                    text_dict = page.get_text("dict")
                    page_text = ""
                    for block in text_dict["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    page_text += span["text"] + " "
                            page_text += "\n"
                
                # Method 3: If still empty, try OCR-style extraction
                if not page_text.strip():
                    logger.warning(f"Page {page_num+1} has no extractable text")
                    page_text = f"[Page {page_num+1}: No extractable text]\n"
                
                page_texts.append(page_text)
                full_text += page_text + "\n"
                
            except Exception as e:
                logger.error(f"Error extracting page {page_num+1}: {e}")
                page_texts.append(f"[Page {page_num+1}: Extraction error]\n")
        
        doc.close()
        
        logger.info(f"Extracted {len(full_text)} characters from {metadata.page_count} pages")
        return full_text, metadata
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")
        # Fallback to PyPDF2 if available
        try:
            import PyPDF2
            return extract_text_with_pypdf2(pdf_path)
        except ImportError:
            raise Exception(f"PyMuPDF failed and PyPDF2 not available: {e}")

def extract_text_with_pypdf2(pdf_path: str) -> Tuple[str, DocumentMetadata]:
    """Fallback extraction using PyPDF2."""
    import PyPDF2
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        metadata = DocumentMetadata(
            path=pdf_path,
            filename=Path(pdf_path).name,
            page_count=len(pdf_reader.pages),
            file_size=Path(pdf_path).stat().st_size
        )
        
        # Extract metadata if available
        if pdf_reader.metadata:
            metadata.title = pdf_reader.metadata.get('/Title', '')
            metadata.author = pdf_reader.metadata.get('/Author', '')
            metadata.subject = pdf_reader.metadata.get('/Subject', '')
            metadata.creator = pdf_reader.metadata.get('/Creator', '')
            metadata.producer = pdf_reader.metadata.get('/Producer', '')
        
        # Extract text
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        logger.info(f"Fallback extraction: {len(text_content)} characters")
        return text_content, metadata

def advanced_text_cleaning(text: str) -> str:
    """
    Advanced text cleaning building on your existing cleaning functions.
    This extends the cleaning you already have in cleaning/text_cleaner.py
    """
    from cleaning import clean_text  # Use your existing cleaning
    
    # Apply your existing cleaning first
    cleaned_text = clean_text(text)
    
    # Additional PyMuPDF-specific cleaning
    additional_patterns = [
        # Remove fitz extraction artifacts
        r'\x0c',  # Form feed characters
        r'cid:\d+',  # CID references
        r'obj\s*<<.*?>>\s*endobj',  # PDF objects
        r'stream\s*.*?\s*endstream',  # PDF streams
        
        # Academic paper specific
        r'^Keywords:.*$',  # Keywords lines
        r'^Abstract\s*$',  # Abstract headers
        r'^References\s*$',  # Reference headers
        r'^\d+\.\s*INTRODUCTION\s*$',  # Numbered sections
        
        # Legal document specific
        r'^WHEREAS,.*$',  # Legal whereas clauses
        r'^Article\s+\d+.*$',  # Legal articles
        r'^Section\s+\d+.*$',  # Legal sections
    ]
    
    for pattern in additional_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Normalize whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    
    return cleaned_text.strip()

def enhanced_chunking(text: str, document_path: str, chunk_size: int = 500, 
                     overlap: int = 50) -> List[TextChunk]:
    """
    Enhanced chunking that builds on your existing chunking but adds Neo4j metadata.
    
    Args:
        text: Cleaned text
        document_path: Path to source document  
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
        
    Returns:
        List of TextChunk objects ready for Neo4j ingestion
    """
    
    # Method 1: Paragraph-based chunking (preferred)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) > 3:  # If we have good paragraph structure
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) > 20:  # Only substantial paragraphs
                chunk = TextChunk(
                    chunk_id=f"{Path(document_path).stem}_para_{i+1:03d}",
                    text=paragraph,
                    document_path=document_path,
                    chunk_index=i,
                    page_start=0,  # TODO: Track page numbers
                    page_end=0,
                    word_count=len(paragraph.split()),
                    char_count=len(paragraph)
                )
                chunks.append(chunk)
        
        if chunks:  # If paragraph chunking worked
            logger.info(f"Created {len(chunks)} paragraph-based chunks")
            return chunks
    
    # Method 2: Fixed-size chunking with overlap
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_words) > 20:  # Only substantial chunks
            chunk = TextChunk(
                chunk_id=f"{Path(document_path).stem}_chunk_{len(chunks)+1:03d}",
                text=chunk_text,
                document_path=document_path,
                chunk_index=len(chunks),
                page_start=0,  # TODO: Track page numbers
                page_end=0,
                word_count=len(chunk_words),
                char_count=len(chunk_text)
            )
            chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} fixed-size chunks with overlap")
    return chunks

def process_pdf_for_neo4j(pdf_path: str) -> Dict[str, Any]:
    """
    Complete PDF processing pipeline for Neo4j ingestion.
    
    Returns structured data ready for Neo4j:
    {
        'document': DocumentMetadata,
        'chunks': List[TextChunk],
        'raw_text': str,
        'cleaned_text': str
    }
    """
    
    logger.info(f"Processing {pdf_path} for Neo4j ingestion...")
    
    try:
        # Step 1: Extract text and metadata
        raw_text, metadata = extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return None
        
        # Step 2: Clean text  
        cleaned_text = advanced_text_cleaning(raw_text)
        
        # Step 3: Create chunks
        chunks = enhanced_chunking(cleaned_text, pdf_path)
        
        if not chunks:
            logger.warning(f"No chunks created from {pdf_path}")
            return None
        
        # Step 4: Return structured data
        result = {
            'document': metadata,
            'chunks': chunks,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'processing_stats': {
                'raw_chars': len(raw_text),
                'cleaned_chars': len(cleaned_text),
                'chunk_count': len(chunks),
                'avg_chunk_size': sum(c.word_count for c in chunks) // len(chunks) if chunks else 0
            }
        }
        
        logger.info(f"Successfully processed {pdf_path}: {len(chunks)} chunks")
        return result
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return None

def batch_process_pdfs(pdf_directory: str = "pdfs", max_files: int = None) -> List[Dict[str, Any]]:
    """
    Process multiple PDFs for Neo4j ingestion.
    
    Args:
        pdf_directory: Directory containing PDFs
        max_files: Maximum number of files to process (None for all)
        
    Returns:
        List of processed document data
    """
    
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        logger.error(f"PDF directory {pdf_directory} not found")
        return []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    processed_documents = []
    failed_files = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        
        try:
            result = process_pdf_for_neo4j(str(pdf_file))
            if result:
                processed_documents.append(result)
                logger.info(f"✅ Successfully processed {pdf_file.name}")
            else:
                failed_files.append(pdf_file.name)
                logger.warning(f"⚠️ Failed to process {pdf_file.name}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file.name}: {e}")
            failed_files.append(pdf_file.name)
    
    logger.info(f"Batch processing complete: {len(processed_documents)} successful, {len(failed_files)} failed")
    
    if failed_files:
        logger.info(f"Failed files: {failed_files}")
    
    return processed_documents

# TODO: Integration functions for your existing pipeline
def migrate_existing_data():
    """
    Migrate your existing processed chunks to the new format.
    This bridges your current ChromaDB data to Neo4j-ready format.
    """
    
    # Load your existing processed chunks
    chunks_file = Path("output/processed_chunks.json")
    if chunks_file.exists():
        import json
        with open(chunks_file, 'r') as f:
            existing_chunks = json.load(f)
        
        logger.info(f"Found {len(existing_chunks)} existing chunks to migrate")
        
        # Convert to Neo4j format
        neo4j_chunks = []
        for chunk in existing_chunks:
            neo4j_chunk = TextChunk(
                chunk_id=chunk['chunk_id'],
                text=chunk['text'],
                document_path=f"pdfs/{chunk['document']}",  # Reconstruct path
                chunk_index=chunk['chunk_index'],
                page_start=0,  # Unknown from existing data
                page_end=0,
                word_count=chunk['word_count'],
                char_count=chunk['char_count']
            )
            neo4j_chunks.append(neo4j_chunk)
        
        logger.info(f"Migrated {len(neo4j_chunks)} chunks to Neo4j format")
        return neo4j_chunks
    
    return []

if __name__ == "__main__":
    # Test the enhanced processor
    print("🔧 Enhanced PDF Processor for Neo4j RAG")
    print("=" * 50)
    
    # Process a few PDFs as test
    results = batch_process_pdfs(max_files=3)
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} documents")
        
        # Show sample data structure
        sample = results[0]
        print(f"\n📄 Sample Document: {sample['document'].filename}")
        print(f"   Pages: {sample['document'].page_count}")
        print(f"   Chunks: {len(sample['chunks'])}")
        print(f"   Author: {sample['document'].author}")
        print(f"   Title: {sample['document'].title}")
        
        print(f"\n📝 Sample Chunk:")
        sample_chunk = sample['chunks'][0]
        print(f"   ID: {sample_chunk.chunk_id}")
        print(f"   Words: {sample_chunk.word_count}")
        print(f"   Text: {sample_chunk.text[:100]}...")
    
    else:
        print("❌ No documents processed successfully") 