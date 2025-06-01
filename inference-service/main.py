from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import json
import logging
import platform
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Inference Service",
    description="Service for running inference with fine-tuned LLM models",
    version="1.0.0"
)

class InferenceRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

class InferenceResponse(BaseModel):
    generated_text: str
    model_id: str
    prompt: str
    generation_time: float
    tokens_generated: int

# Global variables for model caching
loaded_models = {}
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
        logger.info("Using CPU for inference")
    
    return hardware_info

def load_model(model_path: str, model_id: str):
    """Load a model and tokenizer"""
    try:
        hardware_info = detect_hardware()
        device = hardware_info["device"]
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        # Move model to appropriate device
        if device == "mps":
            model = model.to("mps")
        elif device == "cuda":
            model = model.to("cuda")
        
        # Create pipeline for easier inference
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,  # -1 for CPU/MPS
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32
        )
        
        loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": pipe,
            "device": device,
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model {model_id} loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        return False

def unload_model(model_id: str):
    """Unload a model to free memory"""
    if model_id in loaded_models:
        del loaded_models[model_id]
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Model {model_id} unloaded")

def generate_text(model_id: str, prompt: str, max_tokens: int, temperature: float, top_p: float, do_sample: bool) -> Dict:
    """Generate text using the specified model"""
    if model_id not in loaded_models:
        raise ValueError(f"Model {model_id} is not loaded")
    
    model_info = loaded_models[model_id]
    pipe = model_info["pipeline"]
    
    start_time = datetime.now()
    
    try:
        # Generate text
        result = pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pipe.tokenizer.eos_token_id,
            return_full_text=False  # Only return generated text
        )
        
        generated_text = result[0]["generated_text"]
        
        # Calculate generation time
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        # Count tokens (approximate)
        tokens_generated = len(pipe.tokenizer.encode(generated_text))
        
        return {
            "generated_text": generated_text,
            "generation_time": generation_time,
            "tokens_generated": tokens_generated
        }
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Model Inference Service", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    hardware_info = detect_hardware()
    return {
        "status": "healthy",
        "service": "inference-service",
        "hardware": hardware_info,
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/models")
async def get_available_models():
    """Get list of available models for inference"""
    models = []
    
    # Check for trained models
    models_dir = "/app/models"
    if os.path.exists(models_dir):
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(model_path):
                # Check if it's a valid model directory
                config_file = os.path.join(model_path, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        models.append({
                            "id": model_dir,
                            "name": f"Fine-tuned {model_dir}",
                            "type": "fine-tuned",
                            "path": model_path,
                            "loaded": model_dir in loaded_models,
                            "architecture": config.get("architectures", ["unknown"])[0]
                        })
                    except Exception as e:
                        logger.error(f"Error reading model config {model_path}: {e}")
    
    return {"models": models}

@app.post("/models/{model_id}/load")
async def load_model_endpoint(model_id: str):
    """Load a model for inference"""
    if model_id in loaded_models:
        return {"status": "already_loaded", "model_id": model_id}
    
    # Find model path
    model_path = f"/app/models/{model_id}"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Load model in background
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(executor, load_model, model_path, model_id)
    
    if success:
        return {"status": "loaded", "model_id": model_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.delete("/models/{model_id}/unload")
async def unload_model_endpoint(model_id: str):
    """Unload a model to free memory"""
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    unload_model(model_id)
    return {"status": "unloaded", "model_id": model_id}

@app.post("/generate")
async def generate_text_endpoint(request: InferenceRequest):
    """Generate text using a loaded model"""
    try:
        # Check if model is loaded
        if request.model_id not in loaded_models:
            # Try to load the model automatically
            model_path = f"/app/models/{request.model_id}"
            if os.path.exists(model_path):
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(executor, load_model, model_path, request.model_id)
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to load model")
            else:
                raise HTTPException(status_code=404, detail="Model not found")
        
        # Generate text in background
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            generate_text,
            request.model_id,
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.do_sample
        )
        
        return InferenceResponse(
            generated_text=result["generated_text"],
            model_id=request.model_id,
            prompt=request.prompt,
            generation_time=result["generation_time"],
            tokens_generated=result["tokens_generated"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        raise HTTPException(status_code=500, detail="Text generation failed")

@app.post("/batch_generate")
async def batch_generate_text(requests: List[InferenceRequest]):
    """Generate text for multiple requests"""
    results = []
    
    for request in requests:
        try:
            # Check if model is loaded
            if request.model_id not in loaded_models:
                model_path = f"/app/models/{request.model_id}"
                if os.path.exists(model_path):
                    loop = asyncio.get_event_loop()
                    success = await loop.run_in_executor(executor, load_model, model_path, request.model_id)
                    if not success:
                        results.append({"error": "Failed to load model", "request": request.dict()})
                        continue
                else:
                    results.append({"error": "Model not found", "request": request.dict()})
                    continue
            
            # Generate text
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                generate_text,
                request.model_id,
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.do_sample
            )
            
            results.append({
                "generated_text": result["generated_text"],
                "model_id": request.model_id,
                "prompt": request.prompt,
                "generation_time": result["generation_time"],
                "tokens_generated": result["tokens_generated"]
            })
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            results.append({"error": str(e), "request": request.dict()})
    
    return {"results": results}

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    if model_id not in loaded_models:
        # Check if model exists but not loaded
        model_path = f"/app/models/{model_id}"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "model_id": model_id,
            "status": "not_loaded",
            "path": model_path
        }
    
    model_info = loaded_models[model_id]
    return {
        "model_id": model_id,
        "status": "loaded",
        "device": model_info["device"],
        "loaded_at": model_info["loaded_at"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 