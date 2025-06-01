from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import PyPDF2
import fitz  # PyMuPDF
import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Processor Service",
    description="Service for processing PDF files and extracting text for LLM training",
    version="1.0.0"
)

class ProcessRequest(BaseModel):
    file_path: str
    pdf_id: str
    chunk_size: int = 1000
    overlap: int = 200

class ProcessedDocument(BaseModel):
    id: str
    pdf_id: str
    filename: str
    total_pages: int
    total_chunks: int
    processed_at: str
    chunks: List[dict]

def clean_text(text: str) -> str:
    """Clean and normalize extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 10]
    return '\n'.join(lines)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for training"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def extract_text_pymupdf(file_path: str) -> str:
    """Extract text using PyMuPDF (better for complex PDFs)"""
    try:
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        doc.close()
        return text
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        raise

def extract_text_pypdf2(file_path: str) -> str:
    """Extract text using PyPDF2 (fallback method)"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "PDF Processor Service", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pdf-processor"}

@app.post("/process")
async def process_pdf(request: ProcessRequest):
    """Process a PDF file and extract text chunks for training"""
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        logger.info(f"Processing PDF: {request.file_path}")
        
        # Try PyMuPDF first, fallback to PyPDF2
        try:
            raw_text = extract_text_pymupdf(request.file_path)
            extraction_method = "PyMuPDF"
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            raw_text = extract_text_pypdf2(request.file_path)
            extraction_method = "PyPDF2"
        
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Clean the text
        cleaned_text = clean_text(raw_text)
        
        # Split into chunks
        chunks = chunk_text(cleaned_text, request.chunk_size, request.overlap)
        
        # Count pages
        try:
            doc = fitz.open(request.file_path)
            total_pages = doc.page_count
            doc.close()
        except:
            # Fallback page counting
            with open(request.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
        
        # Create processed document structure
        processed_doc = {
            "id": str(uuid.uuid4()),
            "pdf_id": request.pdf_id,
            "filename": os.path.basename(request.file_path),
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "processed_at": datetime.now().isoformat(),
            "extraction_method": extraction_method,
            "chunks": [
                {
                    "id": str(uuid.uuid4()),
                    "index": i,
                    "text": chunk,
                    "length": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        # Save processed data
        output_dir = "/app/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{request.pdf_id}_processed.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, indent=2, ensure_ascii=False)
        
        # Also save as training format (JSONL)
        training_file = os.path.join(output_dir, f"{request.pdf_id}_training.jsonl")
        with open(training_file, 'w', encoding='utf-8') as f:
            for chunk in processed_doc["chunks"]:
                training_example = {
                    "text": chunk["text"],
                    "source": processed_doc["filename"],
                    "chunk_id": chunk["id"]
                }
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully processed PDF: {len(chunks)} chunks extracted")
        
        return {
            "status": "success",
            "pdf_id": request.pdf_id,
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "extraction_method": extraction_method,
            "output_file": output_file,
            "training_file": training_file
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/processed/{pdf_id}")
async def get_processed_data(pdf_id: str):
    """Get processed data for a specific PDF"""
    output_file = f"/app/processed/{pdf_id}_processed.json"
    
    if not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading processed data: {str(e)}")

@app.get("/processed")
async def list_processed_files():
    """List all processed PDF files"""
    processed_dir = "/app/processed"
    if not os.path.exists(processed_dir):
        return {"processed_files": []}
    
    files = []
    for filename in os.listdir(processed_dir):
        if filename.endswith('_processed.json'):
            file_path = os.path.join(processed_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                files.append({
                    "pdf_id": data["pdf_id"],
                    "filename": data["filename"],
                    "total_pages": data["total_pages"],
                    "total_chunks": data["total_chunks"],
                    "processed_at": data["processed_at"]
                })
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
                continue
    
    return {"processed_files": files}

@app.delete("/processed/{pdf_id}")
async def delete_processed_data(pdf_id: str):
    """Delete processed data for a specific PDF"""
    files_to_delete = [
        f"/app/processed/{pdf_id}_processed.json",
        f"/app/processed/{pdf_id}_training.jsonl"
    ]
    
    deleted_files = []
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail="No processed data found to delete")
    
    return {"status": "success", "deleted_files": deleted_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 