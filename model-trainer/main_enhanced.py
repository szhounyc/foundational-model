#!/usr/bin/env python3
"""
Enhanced Model Training Service

This service provides advanced fine-tuning capabilities with LoRA support,
hardware optimization, and comprehensive monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import ray
import torch
import os
import json
import logging
import platform
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Union
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our enhanced components
from config import (
    get_model_config, get_lora_config, get_training_strategy,
    get_hardware_config, adjust_config_for_hardware,
    MODEL_CONFIGS, LORA_CONFIGS, TRAINING_STRATEGIES
)
from data_loader import DatasetLoader, DatasetStatistics
from enhanced_trainer import EnhancedTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Model Training Service",
    description="Advanced LLM fine-tuning service with LoRA support and hardware optimization",
    version="2.0.0"
)

# Pydantic models for API
class TrainingRequest(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    model_name: str = Field(..., description="Model to fine-tune")
    dataset_files: List[str] = Field(..., description="List of dataset files")
    dataset_format: str = Field(default="auto", description="Dataset format (auto, fireworks, alpaca, text)")
    
    # Training configuration
    training_strategy: str = Field(default="balanced", description="Training strategy")
    lora_size: str = Field(default="medium", description="LoRA configuration size")
    use_lora: bool = Field(default=True, description="Whether to use LoRA")
    
    # Hyperparameters (optional overrides)
    learning_rate: Optional[float] = Field(default=None, description="Learning rate override")
    num_epochs: Optional[int] = Field(default=None, description="Number of epochs override")
    batch_size: Optional[int] = Field(default=None, description="Batch size override")
    max_length: Optional[int] = Field(default=None, description="Max sequence length override")
    validation_split: float = Field(default=0.1, description="Validation split ratio")
    
    # Advanced options
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    save_steps: Optional[int] = Field(default=None, description="Save checkpoint every N steps")

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed, cancelled
    progress: float
    current_step: int
    total_steps: int
    current_epoch: int
    total_epochs: int
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    model_path: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    model_id: str
    max_length: int
    recommended_batch_size: int
    parameters: str
    description: str

class DatasetInfo(BaseModel):
    total_examples: int
    avg_length: float
    format_distribution: Dict[str, int]
    length_distribution: Dict[str, int]

# Global variables
training_jobs: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=2)
hardware_info = None

def detect_hardware():
    """Detect available hardware and configure accordingly"""
    global hardware_info
    
    if hardware_info is not None:
        return hardware_info
    
    hardware_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": "cpu",
        "device_count": 1,
        "memory_gb": 0,
        "hardware_type": "cpu"
    }
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_info = result.stdout.strip()
            if "M4" in cpu_info:
                hardware_info.update({
                    "device": "mps",
                    "hardware_type": "apple_m4_max",
                    "is_m4_max": True
                })
                logger.info("Detected Apple M4 Max - using MPS acceleration")
            else:
                hardware_info.update({
                    "device": "mps",
                    "hardware_type": "apple_silicon",
                    "is_m4_max": False
                })
                logger.info(f"Detected Apple Silicon: {cpu_info}")
        except:
            hardware_info["hardware_type"] = "apple_silicon"
    
    # Check for CUDA
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        hardware_info.update({
            "device": "cuda",
            "device_count": device_count,
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "hardware_type": "cuda_multi" if device_count > 1 else "cuda_single"
        })
        logger.info(f"Detected CUDA with {device_count} GPU(s)")
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        hardware_info.update({
            "device": "mps",
            "hardware_type": "apple_silicon"
        })
        logger.info("Detected MPS (Apple Silicon) acceleration")
    
    else:
        logger.info("Using CPU for training")
    
    return hardware_info

async def run_training_job(config: Dict):
    """Run training job with enhanced trainer"""
    job_id = config["job_id"]
    
    try:
        # Update job status
        training_jobs[job_id].update({
            "status": "running",
            "started_at": datetime.now().isoformat()
        })
        
        # Get configurations
        model_config = get_model_config(config["model_name"])
        hardware_type = detect_hardware()["hardware_type"]
        
        # Adjust model config for hardware
        model_config = adjust_config_for_hardware(model_config, hardware_type)
        
        # Apply hyperparameter overrides
        if config.get("learning_rate"):
            model_config.learning_rate = config["learning_rate"]
        if config.get("num_epochs"):
            model_config.num_train_epochs = config["num_epochs"]
        if config.get("batch_size"):
            model_config.batch_size = config["batch_size"]
        if config.get("max_length"):
            model_config.max_length = config["max_length"]
        if config.get("save_steps"):
            model_config.save_steps = config["save_steps"]
        
        # Get training strategy and LoRA config
        training_strategy = get_training_strategy(config.get("training_strategy", "balanced"))
        if not config.get("use_lora", True):
            training_strategy.use_lora = False
        
        lora_config = None
        if training_strategy.use_lora:
            lora_config = get_lora_config(config.get("lora_size", "medium"))
        
        # Create enhanced trainer
        trainer = EnhancedTrainer(
            job_id=job_id,
            model_config=model_config,
            lora_config=lora_config,
            training_strategy=training_strategy,
            hardware_type=hardware_type
        )
        
        # Load model and tokenizer
        trainer.load_model_and_tokenizer()
        
        # Load and prepare dataset
        data_loader = DatasetLoader(trainer.tokenizer, model_config.max_length)
        dataset = data_loader.load_dataset(
            config["dataset_files"], 
            config.get("dataset_format", "auto")
        )
        
        # Analyze dataset
        stats = DatasetStatistics.analyze_dataset(dataset)
        DatasetStatistics.print_statistics(stats)
        
        # Update job with dataset info
        training_jobs[job_id]["dataset_info"] = stats
        
        # Prepare dataset for training
        prepared_dataset = data_loader.prepare_dataset(
            dataset, 
            validation_split=config.get("validation_split", 0.1)
        )
        
        # Start training
        result = trainer.train(prepared_dataset)
        
        # Update job status
        if result["status"] == "completed":
            training_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "model_path": result["output_dir"],
                "final_loss": result.get("train_loss"),
                "train_runtime": result.get("train_runtime"),
                "samples_per_second": result.get("train_samples_per_second")
            })
        else:
            training_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": result.get("error", "Unknown error")
            })
        
        logger.info(f"Training job {job_id} completed with status: {result['status']}")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        training_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize Ray and detect hardware on startup"""
    try:
        if not ray.is_initialized():
            # Initialize Ray with appropriate configuration
            hardware = detect_hardware()
            
            if hardware["device"] == "cuda":
                ray.init(num_gpus=hardware["device_count"])
            else:
                ray.init()
            
            logger.info("Ray initialized successfully")
        
        logger.info(f"Hardware detected: {hardware}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize Ray: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if ray.is_initialized():
        ray.shutdown()
    executor.shutdown(wait=True)

@app.get("/")
async def root():
    return {
        "message": "Enhanced Model Training Service", 
        "version": "2.0.0",
        "features": ["LoRA", "Hardware Optimization", "Advanced Monitoring"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    hardware = detect_hardware()
    return {
        "status": "healthy", 
        "service": "enhanced-model-trainer",
        "ray_initialized": ray.is_initialized(),
        "hardware": hardware,
        "active_jobs": len([j for j in training_jobs.values() if j["status"] == "running"])
    }

@app.post("/train", response_model=Dict[str, str])
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start an enhanced training job"""
    try:
        # Validate model
        if request.model_name not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )
        
        # Validate training strategy
        if request.training_strategy not in TRAINING_STRATEGIES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported strategy: {request.training_strategy}. Available: {list(TRAINING_STRATEGIES.keys())}"
            )
        
        # Validate LoRA size
        if request.lora_size not in LORA_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported LoRA size: {request.lora_size}. Available: {list(LORA_CONFIGS.keys())}"
            )
        
        # Initialize job tracking
        training_jobs[request.job_id] = {
            "job_id": request.job_id,
            "model_name": request.model_name,
            "status": "queued",
            "progress": 0.0,
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": request.num_epochs or 3,
            "created_at": datetime.now().isoformat(),
            "config": request.dict()
        }
        
        # Run training in background
        background_tasks.add_task(run_training_job, request.dict())
        
        return {
            "status": "started",
            "job_id": request.job_id,
            "message": "Enhanced training job queued successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs", response_model=Dict[str, List[TrainingStatus]])
async def get_training_jobs():
    """Get all training jobs"""
    jobs = []
    for job_data in training_jobs.values():
        jobs.append(TrainingStatus(**job_data))
    return {"jobs": jobs}

@app.get("/jobs/{job_id}", response_model=TrainingStatus)
async def get_training_job(job_id: str):
    """Get specific training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return TrainingStatus(**training_jobs[job_id])

@app.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if training_jobs[job_id]["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed job")
    
    # Update status
    training_jobs[job_id].update({
        "status": "cancelled",
        "completed_at": datetime.now().isoformat()
    })
    
    return {"status": "cancelled", "job_id": job_id}

@app.get("/models/available", response_model=Dict[str, List[ModelInfo]])
async def get_available_models():
    """Get list of available models for training"""
    models = []
    
    model_descriptions = {
        "llama-3.2-1b": "Meta Llama 3.2 1B - Fast and efficient for most tasks",
        "llama-3.2-3b": "Meta Llama 3.2 3B - Balanced performance and speed",
        "llama-3.1-8b": "Meta Llama 3.1 8B - High performance for complex tasks",
        "deepseek-coder-1.3b": "DeepSeek Coder 1.3B - Optimized for code generation",
        "deepseek-coder-6.7b": "DeepSeek Coder 6.7B - Advanced code understanding",
        "mistral-7b": "Mistral 7B - Excellent general-purpose model",
        "phi-3-mini": "Microsoft Phi-3 Mini - Compact and efficient",
        "qwen2.5-1.5b": "Qwen 2.5 1.5B - Fast multilingual model",
        "qwen2.5-7b": "Qwen 2.5 7B - Advanced multilingual capabilities"
    }
    
    parameter_counts = {
        "llama-3.2-1b": "1.2B",
        "llama-3.2-3b": "3.2B", 
        "llama-3.1-8b": "8.0B",
        "deepseek-coder-1.3b": "1.3B",
        "deepseek-coder-6.7b": "6.7B",
        "mistral-7b": "7.2B",
        "phi-3-mini": "3.8B",
        "qwen2.5-1.5b": "1.5B",
        "qwen2.5-7b": "7.6B"
    }
    
    for model_name, config in MODEL_CONFIGS.items():
        models.append(ModelInfo(
            id=model_name,
            name=model_name.replace("-", " ").title(),
            model_id=config.model_id,
            max_length=config.max_length,
            recommended_batch_size=config.batch_size,
            parameters=parameter_counts.get(model_name, "Unknown"),
            description=model_descriptions.get(model_name, "No description available")
        ))
    
    return {"models": models}

@app.get("/strategies", response_model=Dict[str, List[str]])
async def get_training_strategies():
    """Get available training strategies"""
    return {
        "strategies": list(TRAINING_STRATEGIES.keys()),
        "lora_sizes": list(LORA_CONFIGS.keys())
    }

@app.get("/hardware")
async def get_hardware_info():
    """Get current hardware information"""
    return detect_hardware()

@app.post("/dataset/analyze")
async def analyze_dataset(dataset_files: List[str], dataset_format: str = "auto"):
    """Analyze dataset without training"""
    try:
        # Create a temporary tokenizer for analysis
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        
        # Load dataset
        data_loader = DatasetLoader(tokenizer, 2048)
        dataset = data_loader.load_dataset(dataset_files, dataset_format)
        
        # Analyze
        stats = DatasetStatistics.analyze_dataset(dataset)
        
        return DatasetInfo(**stats)
        
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 