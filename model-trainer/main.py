from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import ray
from ray import tune
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import os
import json
import logging
import platform
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Training Service",
    description="Service for fine-tuning LLM models using Ray.io",
    version="1.0.0"
)

class TrainingRequest(BaseModel):
    job_id: str
    model_name: str
    dataset_files: List[str]
    hyperparameters: Dict

class TrainingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None

# Global variables for tracking jobs
training_jobs = {}
executor = ThreadPoolExecutor(max_workers=2)

def detect_hardware():
    """Detect available hardware and configure accordingly"""
    hardware_info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": "cpu",
        "device_count": 1,
        "memory_gb": 0
    }
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Check if it's M4 Max specifically
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            cpu_info = result.stdout.strip()
            if "M4" in cpu_info:
                hardware_info["device"] = "mps"  # Metal Performance Shaders
                hardware_info["is_m4_max"] = True
                logger.info("Detected Apple M4 Max - using MPS acceleration")
            else:
                hardware_info["is_m4_max"] = False
                logger.info(f"Detected Apple Silicon: {cpu_info}")
        except:
            hardware_info["is_m4_max"] = False
    
    # Check for CUDA
    elif torch.cuda.is_available():
        hardware_info["device"] = "cuda"
        hardware_info["device_count"] = torch.cuda.device_count()
        hardware_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Detected CUDA with {hardware_info['device_count']} GPU(s)")
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        hardware_info["device"] = "mps"
        logger.info("Detected MPS (Apple Silicon) acceleration")
    
    else:
        logger.info("Using CPU for training")
    
    return hardware_info

def get_model_config(model_name: str) -> Dict:
    """Get model configuration based on model name"""
    configs = {
        "llama-3.2-1b": {
            "model_id": "meta-llama/Llama-3.2-1B",
            "max_length": 2048,
            "batch_size": 4,
            "gradient_accumulation_steps": 4
        },
        "llama-3.2-3b": {
            "model_id": "meta-llama/Llama-3.2-3B",
            "max_length": 2048,
            "batch_size": 2,
            "gradient_accumulation_steps": 8
        },
        "deepseek-coder-1.3b": {
            "model_id": "deepseek-ai/deepseek-coder-1.3b-base",
            "max_length": 2048,
            "batch_size": 4,
            "gradient_accumulation_steps": 4
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-v0.1",
            "max_length": 2048,
            "batch_size": 1,
            "gradient_accumulation_steps": 16
        },
        "phi-3-mini": {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "max_length": 2048,
            "batch_size": 4,
            "gradient_accumulation_steps": 4
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return configs[model_name]

def load_training_data(dataset_files: List[str]) -> Dataset:
    """Load and prepare training data from processed PDF files"""
    all_texts = []
    
    for file_path in dataset_files:
        full_path = f"/app/data/{file_path}"
        if not os.path.exists(full_path):
            logger.warning(f"Dataset file not found: {full_path}")
            continue
            
        try:
            if file_path.endswith('.jsonl'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if 'text' in data:
                            all_texts.append(data['text'])
            elif file_path.endswith('.json'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'chunks' in data:
                        for chunk in data['chunks']:
                            all_texts.append(chunk['text'])
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if not all_texts:
        raise ValueError("No training data could be loaded")
    
    logger.info(f"Loaded {len(all_texts)} training examples")
    return Dataset.from_dict({"text": all_texts})

def train_model_function(config: Dict):
    """Training function to be executed by Ray"""
    try:
        # Get hardware info
        hardware_info = detect_hardware()
        device = hardware_info["device"]
        
        # Load model configuration
        model_config = get_model_config(config["model_name"])
        model_id = model_config["model_id"]
        
        # Load tokenizer and model
        logger.info(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Move model to appropriate device
        if device == "mps":
            model = model.to("mps")
        elif device == "cuda":
            model = model.to("cuda")
        
        # Load and tokenize data
        dataset = load_training_data(config["dataset_files"])
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=model_config["max_length"]
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"/app/checkpoints/{config['job_id']}",
            overwrite_output_dir=True,
            num_train_epochs=config["hyperparameters"].get("num_epochs", 3),
            per_device_train_batch_size=config["hyperparameters"].get("batch_size", model_config["batch_size"]),
            gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
            learning_rate=config["hyperparameters"].get("learning_rate", 2e-5),
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Disable for MPS compatibility
            fp16=device == "cuda",  # Only use fp16 with CUDA
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Custom trainer with progress tracking
        class CustomTrainer(Trainer):
            def __init__(self, job_id, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.job_id = job_id
            
            def log(self, logs):
                super().log(logs)
                # Update job status
                if self.job_id in training_jobs:
                    training_jobs[self.job_id].update({
                        "progress": (self.state.epoch / self.args.num_train_epochs) * 100,
                        "current_epoch": int(self.state.epoch),
                        "loss": logs.get("train_loss"),
                        "learning_rate": logs.get("learning_rate")
                    })
        
        # Create trainer
        trainer = CustomTrainer(
            job_id=config["job_id"],
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info(f"Starting training for job {config['job_id']}")
        training_jobs[config["job_id"]]["status"] = "training"
        
        trainer.train()
        
        # Save the final model
        model_save_path = f"/app/models/{config['job_id']}"
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        # Update job status
        training_jobs[config["job_id"]].update({
            "status": "completed",
            "progress": 100.0,
            "completed_at": datetime.now().isoformat(),
            "model_path": model_save_path
        })
        
        logger.info(f"Training completed for job {config['job_id']}")
        return {"status": "success", "model_path": model_save_path}
        
    except Exception as e:
        logger.error(f"Training failed for job {config['job_id']}: {e}")
        training_jobs[config["job_id"]].update({
            "status": "failed",
            "error": str(e)
        })
        raise

# Initialize Ray
@app.on_event("startup")
async def startup_event():
    """Initialize Ray on startup"""
    try:
        if not ray.is_initialized():
            # Configure Ray based on hardware
            hardware_info = detect_hardware()
            
            ray_config = {
                "ignore_reinit_error": True,
                "log_to_driver": False
            }
            
            # Configure for Apple M4 Max
            if hardware_info.get("is_m4_max"):
                ray_config.update({
                    "num_cpus": 12,  # M4 Max has 12 CPU cores
                    "num_gpus": 0,   # Ray doesn't directly support MPS
                })
            elif hardware_info["device"] == "cuda":
                ray_config.update({
                    "num_gpus": hardware_info["device_count"]
                })
            
            ray.init(**ray_config)
            logger.info("Ray initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown Ray on app shutdown"""
    if ray.is_initialized():
        ray.shutdown()

@app.get("/")
async def root():
    return {"message": "Model Training Service", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    hardware_info = detect_hardware()
    return {
        "status": "healthy", 
        "service": "model-trainer",
        "ray_initialized": ray.is_initialized(),
        "hardware": hardware_info
    }

@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a model training job"""
    try:
        # Validate model
        model_config = get_model_config(request.model_name)
        
        # Initialize job tracking
        training_jobs[request.job_id] = {
            "job_id": request.job_id,
            "model_name": request.model_name,
            "status": "queued",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": request.hyperparameters.get("num_epochs", 3),
            "started_at": datetime.now().isoformat()
        }
        
        # Submit training job to Ray
        config = {
            "job_id": request.job_id,
            "model_name": request.model_name,
            "dataset_files": request.dataset_files,
            "hyperparameters": request.hyperparameters
        }
        
        # Run training in background
        background_tasks.add_task(run_training_job, config)
        
        return {
            "status": "started",
            "job_id": request.job_id,
            "message": "Training job queued successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_training_job(config: Dict):
    """Run training job in background"""
    try:
        # Use Ray to run the training
        if ray.is_initialized():
            # Create a Ray task
            @ray.remote
            def ray_train_task(config):
                return train_model_function(config)
            
            # Submit the task
            future = ray_train_task.remote(config)
            result = ray.get(future)
        else:
            # Fallback to direct execution
            result = train_model_function(config)
            
    except Exception as e:
        logger.error(f"Training job failed: {e}")
        training_jobs[config["job_id"]].update({
            "status": "failed",
            "error": str(e)
        })

@app.get("/jobs")
async def get_training_jobs():
    """Get all training jobs"""
    return {"jobs": list(training_jobs.values())}

@app.get("/jobs/{job_id}")
async def get_training_job(job_id: str):
    """Get specific training job status"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]

@app.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Update status
    training_jobs[job_id]["status"] = "cancelled"
    
    return {"status": "cancelled", "job_id": job_id}

@app.get("/models/available")
async def get_available_models():
    """Get list of available models for training"""
    models = []
    for model_name in ["llama-3.2-1b", "llama-3.2-3b", "deepseek-coder-1.3b", "mistral-7b", "phi-3-mini"]:
        config = get_model_config(model_name)
        models.append({
            "id": model_name,
            "name": model_name,
            "model_id": config["model_id"],
            "max_length": config["max_length"],
            "recommended_batch_size": config["batch_size"]
        })
    
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 