from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import sqlite3
from contextlib import asynccontextmanager
import aiofiles
import PyPDF2
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database models
class TrainingJob(BaseModel):
    id: str
    model_name: str
    dataset_files: List[str]
    status: str
    created_at: str
    completed_at: Optional[str] = None
    hyperparameters: dict
    progress: float = 0.0

class InferenceRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

class ContractReviewRequest(BaseModel):
    contract_files: List[str]  # List of file IDs
    model_id: str = "sftj-s1xkr35z"
    review_type: str = "comprehensive"

class ModelInfo(BaseModel):
    id: str
    name: str
    type: str
    status: str
    size_mb: float
    created_at: str

# Database initialization
def init_db():
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    
    # Training jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_jobs (
            id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            dataset_files TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            hyperparameters TEXT NOT NULL,
            progress REAL DEFAULT 0.0
        )
    ''')
    
    # Models table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            size_mb REAL NOT NULL,
            created_at TEXT NOT NULL,
            file_path TEXT
        )
    ''')
    
    # PDF files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_files (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            processed BOOLEAN DEFAULT FALSE,
            uploaded_at TEXT NOT NULL,
            size_bytes INTEGER,
            extracted_text TEXT
        )
    ''')
    
    # Contract reviews table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contract_reviews (
            id TEXT PRIMARY KEY,
            contract_files TEXT NOT NULL,
            model_id TEXT NOT NULL,
            review_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            review_result TEXT,
            processing_time REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    logger.info(f"Starting PDF text extraction for: {file_path}")
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            page_count = len(pdf_reader.pages)
            logger.info(f"PDF has {page_count} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                logger.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            
        total_chars = len(text.strip())
        logger.info(f"Successfully extracted {total_chars} characters from PDF: {file_path}")
        
        # Log first 500 characters for debugging
        preview_text = text.strip()[:500]
        logger.info(f"PDF content preview (first 500 chars): {preview_text}...")
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    yield
    # Shutdown

app = FastAPI(
    title="LLM Fine-tuning Platform API",
    description="Backend API for fine-tuning and inference with LLM models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000", "http://localhost:9000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
PDF_PROCESSOR_URL = os.getenv("PDF_PROCESSOR_URL", "http://localhost:8001")
TRAINER_URL = os.getenv("TRAINER_URL", "http://localhost:8002")
INFERENCE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://host.docker.internal:9200")

@app.get("/")
async def root():
    return {"message": "LLM Fine-tuning Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = {}
    
    async with httpx.AsyncClient() as client:
        # Check PDF processor
        try:
            response = await client.get(f"{PDF_PROCESSOR_URL}/health", timeout=5.0)
            services_status["pdf_processor"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            services_status["pdf_processor"] = "unreachable"
        
        # Check trainer
        try:
            response = await client.get(f"{TRAINER_URL}/health", timeout=5.0)
            services_status["trainer"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            services_status["trainer"] = "unreachable"
        
        # Check inference service
        try:
            response = await client.get(f"{INFERENCE_URL}/health", timeout=5.0)
            services_status["inference"] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            services_status["inference"] = "unreachable"
    
    return {"status": "healthy", "services": services_status}

# PDF Management Endpoints
@app.post("/api/pdfs/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id = str(uuid.uuid4())
    file_path = f"data/pdfs/{file_id}_{file.filename}"
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Save to database
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO pdf_files (id, filename, file_path, uploaded_at, size_bytes)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_id, file.filename, file_path, datetime.now().isoformat(), len(content)))
    conn.commit()
    conn.close()
    
    return {"id": file_id, "filename": file.filename, "status": "uploaded"}

@app.get("/api/pdfs")
async def list_pdfs():
    """List all uploaded PDF files"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pdf_files ORDER BY uploaded_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    pdfs = []
    for row in rows:
        pdfs.append({
            "id": row[0],
            "filename": row[1],
            "file_path": row[2],
            "processed": bool(row[3]),
            "uploaded_at": row[4],
            "size_bytes": row[5]
        })
    
    return {"pdfs": pdfs}

@app.post("/api/pdfs/{pdf_id}/process")
async def process_pdf(pdf_id: str, background_tasks: BackgroundTasks):
    """Process a PDF file to extract text for training"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT file_path FROM pdf_files WHERE id = ?', (pdf_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    file_path = result[0]
    
    # Call PDF processor service
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{PDF_PROCESSOR_URL}/process",
                json={"file_path": file_path, "pdf_id": pdf_id}
            )
            if response.status_code == 200:
                # Update database
                conn = sqlite3.connect('app.db')
                cursor = conn.cursor()
                cursor.execute('UPDATE pdf_files SET processed = TRUE WHERE id = ?', (pdf_id,))
                conn.commit()
                conn.close()
                
                return {"status": "processing", "pdf_id": pdf_id}
            else:
                raise HTTPException(status_code=500, detail="PDF processing failed")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="PDF processor service unavailable")

# Model Management Endpoints
@app.get("/api/models/available")
async def get_available_models():
    """Get list of available models for fine-tuning"""
    models = [
        {"id": "llama-3.2-1b", "name": "Llama 3.2 1B", "type": "llama", "size": "1B parameters"},
        {"id": "llama-3.2-3b", "name": "Llama 3.2 3B", "type": "llama", "size": "3B parameters"},
        {"id": "deepseek-coder-1.3b", "name": "DeepSeek Coder 1.3B", "type": "deepseek", "size": "1.3B parameters"},
        {"id": "mistral-7b", "name": "Mistral 7B", "type": "mistral", "size": "7B parameters"},
        {"id": "phi-3-mini", "name": "Phi-3 Mini", "type": "phi", "size": "3.8B parameters"},
    ]
    return {"models": models}

@app.get("/api/models/trained")
async def get_trained_models():
    """Get list of trained/fine-tuned models"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM models ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    models = []
    for row in rows:
        models.append({
            "id": row[0],
            "name": row[1],
            "type": row[2],
            "status": row[3],
            "size_mb": row[4],
            "created_at": row[5],
            "file_path": row[6]
        })
    
    return {"models": models}

# Training Endpoints
@app.post("/api/training/start")
async def start_training(
    model_name: str,
    dataset_files: List[str],
    hyperparameters: dict = None
):
    """Start a fine-tuning job"""
    if hyperparameters is None:
        hyperparameters = {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 3,
            "max_length": 512
        }
    
    job_id = str(uuid.uuid4())
    
    # Save job to database
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO training_jobs (id, model_name, dataset_files, status, created_at, hyperparameters)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        job_id,
        model_name,
        json.dumps(dataset_files),
        "queued",
        datetime.now().isoformat(),
        json.dumps(hyperparameters)
    ))
    conn.commit()
    conn.close()
    
    # Call trainer service
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{TRAINER_URL}/train",
                json={
                    "job_id": job_id,
                    "model_name": model_name,
                    "dataset_files": dataset_files,
                    "hyperparameters": hyperparameters
                }
            )
            if response.status_code == 200:
                return {"job_id": job_id, "status": "started"}
            else:
                raise HTTPException(status_code=500, detail="Training service error")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Training service unavailable")

@app.get("/api/training/jobs")
async def get_training_jobs():
    """Get all training jobs"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM training_jobs ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    jobs = []
    for row in rows:
        jobs.append({
            "id": row[0],
            "model_name": row[1],
            "dataset_files": json.loads(row[2]),
            "status": row[3],
            "created_at": row[4],
            "completed_at": row[5],
            "hyperparameters": json.loads(row[6]),
            "progress": row[7]
        })
    
    return {"jobs": jobs}

@app.get("/api/training/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get specific training job details"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM training_jobs WHERE id = ?', (job_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return {
        "id": row[0],
        "model_name": row[1],
        "dataset_files": json.loads(row[2]),
        "status": row[3],
        "created_at": row[4],
        "completed_at": row[5],
        "hyperparameters": json.loads(row[6]),
        "progress": row[7]
    }

# Inference Endpoints
@app.post("/api/inference/generate")
async def generate_text(request: InferenceRequest):
    """Generate text using a trained model"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{INFERENCE_URL}/generate",
                json=request.dict()
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=500, detail="Inference service error")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Inference service unavailable")

@app.get("/api/inference/models")
async def get_inference_models():
    """Get available models for inference"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{INFERENCE_URL}/models")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=500, detail="Inference service error")
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Inference service unavailable")

# Contract Review Endpoints
@app.post("/api/contracts/upload")
async def upload_contract(file: UploadFile = File(...)):
    """Upload a contract PDF file for review"""
    logger.info(f"Starting contract upload for file: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        logger.error(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_id = str(uuid.uuid4())
    file_path = f"data/pdfs/{file_id}_{file.filename}"
    
    logger.info(f"Saving file to: {file_path}")
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    logger.info(f"File saved successfully, size: {len(content)} bytes")
    
    # Extract text from PDF
    try:
        logger.info(f"Starting text extraction for uploaded file: {file.filename}")
        extracted_text = extract_text_from_pdf(file_path)
        text_length = len(extracted_text)
        logger.info(f"Text extraction successful. Extracted {text_length} characters")
        
        # Log text content summary
        if extracted_text:
            lines = extracted_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            logger.info(f"Text extraction summary:")
            logger.info(f"  - Total lines: {len(lines)}")
            logger.info(f"  - Non-empty lines: {len(non_empty_lines)}")
            logger.info(f"  - Average line length: {sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0:.1f}")
            
            # Log first few lines for debugging
            preview_lines = non_empty_lines[:5]
            logger.info(f"First few lines of extracted text:")
            for i, line in enumerate(preview_lines):
                logger.info(f"  Line {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
        
    except Exception as e:
        logger.error(f"Text extraction failed for {file.filename}: {str(e)}")
        # If text extraction fails, still save the file but mark as not processed
        extracted_text = None
        text_length = 0

    # Save to database
    logger.info(f"Saving file metadata to database")
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO pdf_files (id, filename, file_path, uploaded_at, size_bytes, extracted_text, processed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (file_id, file.filename, file_path, datetime.now().isoformat(), len(content), extracted_text, extracted_text is not None))
    conn.commit()
    conn.close()
    
    logger.info(f"Contract upload completed successfully for {file.filename}")

    return {
        "id": file_id, 
        "filename": file.filename, 
        "status": "uploaded",
        "text_extracted": extracted_text is not None,
        "text_length": text_length
    }

@app.get("/api/contracts/files")
async def list_contract_files():
    """Get all uploaded contract files"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, uploaded_at, size_bytes, extracted_text, processed FROM pdf_files ORDER BY uploaded_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    files = []
    for row in rows:
        files.append({
            "id": row[0],
            "filename": row[1],
            "uploaded_at": row[2],
            "size_bytes": row[3],
            "text_extracted": row[4] is not None,
            "text_length": len(row[4]) if row[4] else 0
        })
    
    return {"files": files}

@app.post("/api/contracts/review")
async def review_contracts(request: ContractReviewRequest):
    """Review contract files using the fine-tuned model"""
    review_id = str(uuid.uuid4())
    
    logger.info(f"Starting contract review {review_id}")
    logger.info(f"Request details: files={request.contract_files}, model={request.model_id}, type={request.review_type}")
    
    # Save review request to database
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO contract_reviews (id, contract_files, model_id, review_type, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (review_id, json.dumps(request.contract_files), request.model_id, request.review_type, "processing", datetime.now().isoformat()))
    conn.commit()
    
    try:
        # Get contract texts
        contract_sections = []
        logger.info(f"Retrieving contract texts for {len(request.contract_files)} files")
        
        for i, file_id in enumerate(request.contract_files):
            logger.info(f"Processing contract file {i+1}/{len(request.contract_files)}: {file_id}")
            cursor.execute('SELECT extracted_text, filename FROM pdf_files WHERE id = ?', (file_id,))
            row = cursor.fetchone()
            
            if row and row[0]:
                filename = row[1]
                text_content = row[0]
                contract_text = f"Contract: {filename}\n\n{text_content}"
                contract_sections.append(contract_text)
                
                logger.info(f"Retrieved contract text for {filename}")
                logger.info(f"  - Text length: {len(text_content)} characters")
                logger.info(f"  - Preview: {text_content[:200]}...")
                
            else:
                logger.error(f"Contract file {file_id} not found or text not extracted")
                raise HTTPException(status_code=400, detail=f"Contract file {file_id} not found or text not extracted")
        
        conn.close()
        
        total_text_length = sum(len(section) for section in contract_sections)
        logger.info(f"Prepared {len(contract_sections)} contract sections for review")
        logger.info(f"Total text length: {total_text_length} characters")
        
        # Call inference service for contract review
        logger.info(f"Calling inference service at {INFERENCE_URL}/contract/review")
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            start_time = datetime.now()
            
            inference_request = {
                "contract_sections": contract_sections,
                "model_id": request.model_id,
                "review_type": request.review_type
            }
            
            logger.info(f"Sending inference request:")
            logger.info(f"  - Model ID: {request.model_id}")
            logger.info(f"  - Review type: {request.review_type}")
            logger.info(f"  - Number of sections: {len(contract_sections)}")
            
            response = await client.post(
                f"{INFERENCE_URL}/contract/review",
                json=inference_request
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Inference service response received in {processing_time:.2f} seconds")
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                review_result = response.json()
                logger.info(f"Contract review completed successfully")
                logger.info(f"Review result keys: {list(review_result.keys())}")
                
                # Update database with results
                conn = sqlite3.connect('app.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE contract_reviews 
                    SET status = ?, completed_at = ?, review_result = ?, processing_time = ?
                    WHERE id = ?
                ''', ("completed", end_time.isoformat(), json.dumps(review_result), processing_time, review_id))
                conn.commit()
                conn.close()
                
                logger.info(f"Contract review {review_id} completed and saved to database")
                
                return {
                    "review_id": review_id,
                    "status": "completed",
                    "review": review_result,
                    "processing_time": processing_time
                }
            else:
                logger.error(f"Inference service returned error: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                
                # Update database with error
                conn = sqlite3.connect('app.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE contract_reviews 
                    SET status = ?, completed_at = ?
                    WHERE id = ?
                ''', ("failed", datetime.now().isoformat(), review_id))
                conn.commit()
                conn.close()
                
                raise HTTPException(status_code=500, detail="Contract review failed")
                
    except httpx.RequestError as e:
        logger.error(f"Request error when calling inference service: {str(e)}")
        # Update database with error
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE contract_reviews 
            SET status = ?, completed_at = ?
            WHERE id = ?
        ''', ("failed", datetime.now().isoformat(), review_id))
        conn.commit()
        conn.close()
        
        raise HTTPException(status_code=503, detail="Inference service unavailable")

@app.get("/api/contracts/reviews")
async def get_contract_reviews():
    """Get all contract review history"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM contract_reviews ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    reviews = []
    for row in rows:
        review_result = json.loads(row[7]) if row[7] else None
        reviews.append({
            "id": row[0],
            "contract_files": json.loads(row[1]),
            "model_id": row[2],
            "review_type": row[3],
            "status": row[4],
            "created_at": row[5],
            "completed_at": row[6],
            "review_result": review_result,
            "processing_time": row[8]
        })
    
    return {"reviews": reviews}

@app.get("/api/contracts/reviews/{review_id}")
async def get_contract_review(review_id: str):
    """Get specific contract review details"""
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM contract_reviews WHERE id = ?', (review_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Contract review not found")
    
    review_result = json.loads(row[7]) if row[7] else None
    return {
        "id": row[0],
        "contract_files": json.loads(row[1]),
        "model_id": row[2],
        "review_type": row[3],
        "status": row[4],
        "created_at": row[5],
        "completed_at": row[6],
        "review_result": review_result,
        "processing_time": row[8]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9100) 