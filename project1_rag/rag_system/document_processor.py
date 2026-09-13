"""
document_processor.py
----------------------
Handles loading and chunking documents into pieces the RAG system can search.
Supports PDF, DOCX, and plain TXT files.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any

# For reading PDFs
from pypdf import PdfReader

# For reading Word documents
from docx import Document


def load_document(file_path: str) -> str:
    """
    Load a document and return its text content.
    Supports .pdf, .docx, and .txt files.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext == ".docx":
        return _load_docx(file_path)
    elif ext == ".txt":
        return _load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")


def _load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def _load_docx(file_path: str) -> str:
    """Extract text from a Word document."""
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def _load_txt(file_path: str) -> str:
    """Read a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_name: str = "document"
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks so we don't cut context in half.

    Args:
        text: The full document text
        chunk_size: How many characters per chunk (500 is a good start)
        chunk_overlap: How many characters to repeat between chunks (for context continuity)
        source_name: The document name, used for citations later

    Returns:
        List of dicts with 'text', 'source', and 'chunk_id' keys
    """
    # Clean up the text a bit
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive blank lines
    text = text.strip()

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        # Find the end of this chunk
        end = start + chunk_size

        # If we're not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Try to find a sentence ending near the chunk boundary
            sentence_end = text.rfind('. ', start, end)
            if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                end = sentence_end + 1  # Include the period
            else:
                # Fall back to word boundary
                word_end = text.rfind(' ', start, end)
                if word_end != -1:
                    end = word_end

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:  # Only add non-empty chunks
            chunks.append({
                "text": chunk_text_content,
                "source": source_name,
                "chunk_id": f"{source_name}_{chunk_id}",
                "start_char": start,
                "end_char": end
            })
            chunk_id += 1

        # Move start forward, keeping some overlap
        start = end - chunk_overlap
        if start >= end:  # Safety check to avoid infinite loop
            break

    return chunks


def process_documents(file_paths: List[str], chunk_size: int = 500) -> List[Dict[str, Any]]:
    """
    Process multiple documents and return all chunks combined.
    This is the main function you'll call to prepare documents for the RAG system.
    """
    all_chunks = []

    for file_path in file_paths:
        print(f"📄 Processing: {file_path}")
        try:
            # Load the document
            text = load_document(file_path)
            source_name = Path(file_path).name

            # Split into chunks
            chunks = chunk_text(text, chunk_size=chunk_size, source_name=source_name)
            all_chunks.extend(chunks)

            print(f"   ✅ Created {len(chunks)} chunks from {source_name}")
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")

    print(f"\n📦 Total chunks created: {len(all_chunks)}")
    return all_chunks
