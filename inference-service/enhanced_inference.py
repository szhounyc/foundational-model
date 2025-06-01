#!/usr/bin/env python3
"""
Enhanced Inference Service for Fireworks Models

This service provides advanced inference capabilities for Fireworks fine-tuned models,
using hosted Fireworks API endpoints for inference.
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import asyncio
from dotenv import load_dotenv

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Get API keys from environment
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    # Try loading from current directory as fallback
    load_dotenv()
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    
if not FIREWORKS_API_KEY:
    logger.error("❌ FIREWORKS_API_KEY not found in environment variables")
    logger.error("Please ensure the .env file contains FIREWORKS_API_KEY=your_api_key")
    raise ValueError("FIREWORKS_API_KEY is required")

logger.info(f"✅ FIREWORKS_API_KEY loaded: {FIREWORKS_API_KEY[:10]}...")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import platform

app = FastAPI(
    title="Enhanced Model Inference Service",
    description="Advanced inference service for Fireworks fine-tuned models using hosted API",
    version="3.0.0"
)

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class InferenceRequest(BaseModel):
    model_id: str = Field(..., description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="Chat messages in conversation format")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    top_k: int = Field(default=50, description="Top-k sampling")
    presence_penalty: float = Field(default=0, description="Presence penalty")
    frequency_penalty: float = Field(default=0, description="Frequency penalty")

class InferenceResponse(BaseModel):
    generated_text: str
    model_id: str
    conversation_id: Optional[str] = None
    generation_time: float
    tokens_generated: int
    finish_reason: str

class ContractReviewRequest(BaseModel):
    contract_sections: List[str] = Field(..., description="Contract sections to review")
    model_id: str = Field(default="sftj-qbplmzw9", description="Model to use for review")
    review_type: str = Field(default="comprehensive", description="Type of review: comprehensive, risk_analysis, or quick")

class BatchInferenceRequest(BaseModel):
    requests: List[InferenceRequest]
    max_concurrent: int = Field(default=3, description="Maximum concurrent requests")

class FireworksAPIClient:
    """Client for Fireworks API hosted model inference"""
    
    def __init__(self):
        self.api_key = FIREWORKS_API_KEY
        self.base_url = "https://api.fireworks.ai/inference/v1"
        self.models_dir = Path("/app/models")
        
        # Model mapping from local IDs to hosted deployment IDs
        self.model_mapping = {
            "sftj-qbplmzw9": "accounts/zhsy2011-05908b/deployedModels/zlg-re-fm-sl-mntn-llama3p2-1b-g04rbi28",
            "sftj-s1xkr35z": "accounts/zhsy2011-05908b/deployedModels/zlg-re-fm-sl-mntn-llama3p2-1b-g04rbi28",
            "sftj-wst3swj0": "accounts/zhsy2011-05908b/deployedModels/zlg-re-fm-sl-mntn-llama3p2-1b-g04rbi28",
            # Add more mappings as needed
        }
        
        logger.info(f"✅ Fireworks API Client initialized with {len(self.model_mapping)} model mappings")

    def find_fireworks_models(self) -> List[Dict]:
        """Find all available Fireworks models (from local directory structure)"""
        logger.info("🔍 Searching for Fireworks models...")
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return models
        
        # Find all checkpoint directories
        checkpoint_dirs = []
        for root, dirs, files in os.walk(self.models_dir):
            if "checkpoint" in root and any(f.endswith('.safetensors') for f in files):
                checkpoint_dirs.append(root)
        
        logger.info(f"Found {len(checkpoint_dirs)} potential checkpoint directories")
        
        seen_models = set()
        for checkpoint_path in checkpoint_dirs:
            checkpoint_path = Path(checkpoint_path)
            
            # Generate model ID from path structure
            path_parts = checkpoint_path.parts
            model_id = None
            
            for i, part in enumerate(path_parts):
                if part.startswith('sftj-'):
                    model_id = part
                    break
            
            if not model_id or model_id in seen_models:
                continue
                
            seen_models.add(model_id)
            
            # Check if we have a hosted deployment for this model
            hosted_model = self.model_mapping.get(model_id)
            if not hosted_model:
                logger.warning(f"⚠️ No hosted deployment found for model: {model_id}")
                continue
            
            model_info = {
                "id": model_id,
                "name": f"Fireworks Fine-tuned Model ({model_id})",
                "type": "fireworks_hosted",
                "base_model": "meta-llama/Llama-3.2-1B-Instruct",
                "hosted_deployment": hosted_model,
                "adapter_path": str(checkpoint_path),
                "created_at": datetime.now().isoformat(),
                "status": "available"
            }
            models.append(model_info)
            logger.info(f"✅ Added hosted Fireworks model: {model_id}")
        
        logger.info(f"📊 Total hosted models available: {len(models)}")
        return models

    def call_fireworks_api(self, model_id: str, messages: List[Dict], **kwargs) -> Dict:
        """Call the Fireworks API for inference"""
        
        # Get the hosted deployment ID
        hosted_model = self.model_mapping.get(model_id)
        if not hosted_model:
            raise ValueError(f"No hosted deployment found for model: {model_id}")
        
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": hosted_model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        logger.info(f"🚀 Calling Fireworks API for model: {model_id}")
        logger.info(f"📡 Hosted deployment: {hosted_model}")
        logger.info(f"💬 Messages: {len(messages)} messages")
        
        start_time = datetime.now()
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            result = response.json()
            
            # Extract the generated text
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
                finish_reason = result["choices"][0].get("finish_reason", "stop")
                
                # Estimate token count
                tokens_generated = len(generated_text.split())
                
                logger.info(f"✅ API call successful:")
                logger.info(f"  - Generation time: {generation_time:.2f} seconds")
                logger.info(f"  - Generated text length: {len(generated_text)} characters")
                logger.info(f"  - Estimated tokens: {tokens_generated}")
                logger.info(f"  - Finish reason: {finish_reason}")
                
                return {
                    "generated_text": generated_text,
                    "generation_time": generation_time,
                    "tokens_generated": tokens_generated,
                    "finish_reason": finish_reason,
                    "raw_response": result
                }
            else:
                raise ValueError("Invalid response format from Fireworks API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Fireworks API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            raise HTTPException(status_code=500, detail=f"Fireworks API error: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Initialize Fireworks API client
fireworks_client = FireworksAPIClient()

def generate_contract_review_with_api(contract_sections: List[str], model_id: str) -> str:
    """Generate a contract review using the Fireworks API"""
    
    logger.info(f"Starting contract review with hosted model {model_id}")
    logger.info(f"Input: {len(contract_sections)} contract sections")
    
    # Prepare the contract text
    contract_text = "\n\n".join(contract_sections)
    
    # Log input details
    logger.info(f"Contract review input details:")
    logger.info(f"  - Total sections: {len(contract_sections)}")
    logger.info(f"  - Total text length: {len(contract_text)} characters")
    logger.info(f"  - Text preview (first 500 chars): {contract_text[:500]}...")
    
    # Create a comprehensive prompt for contract review
    system_prompt = """You are an expert contract review attorney with extensive experience in real estate transactions, commercial agreements, and legal document analysis. Your task is to provide a comprehensive, professional contract review that identifies potential risks, issues, and recommendations.

Please analyze the following contract(s) and provide:
1. Executive Summary
2. Key Risk Assessment (High/Medium/Low risks)
3. Specific Issues and Concerns
4. Recommendations for Improvement
5. Legal and Financial Implications

Be thorough, professional, and provide actionable insights."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please review the following contract sections:\n\n{contract_text}"}
    ]
    
    try:
        result = fireworks_client.call_fireworks_api(
            model_id=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7
        )
        
        return result["generated_text"]
        
    except Exception as e:
        logger.error(f"Error generating contract review: {e}")
        raise

def generate_chat_response_with_api(
    model_id: str, 
    messages: List[ChatMessage], 
    **kwargs
) -> Dict:
    """Generate chat response using Fireworks API"""
    
    # Convert ChatMessage objects to dict format
    api_messages = []
    for msg in messages:
        api_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    logger.info(f"Generating chat response with hosted model {model_id}")
    logger.info(f"Input: {len(api_messages)} messages")
    
    try:
        result = fireworks_client.call_fireworks_api(
            model_id=model_id,
            messages=api_messages,
            **kwargs
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Enhanced Model Inference Service", 
        "version": "3.0.0",
        "features": ["fireworks_models", "hosted_inference", "contract_review"],
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    models = fireworks_client.find_fireworks_models()
    return {
        "status": "healthy",
        "service": "enhanced-inference-service",
        "loaded_models": [m["id"] for m in models],
        "model_count": len(models)
    }

@app.get("/models")
async def get_available_models():
    """Get list of available Fireworks models"""
    models = fireworks_client.find_fireworks_models()
    
    return {
        "models": models,
        "total_count": len(models),
        "loaded_count": len([m for m in models if m["type"] == "fireworks_hosted"])
    }

@app.post("/chat/completions")
async def chat_completions(request: InferenceRequest):
    """Generate chat completion using Fireworks API"""
    try:
        logger.info(f"Chat completion request received:")
        logger.info(f"  - Model ID: {request.model_id}")
        logger.info(f"  - Number of messages: {len(request.messages)}")
        logger.info(f"  - Max tokens: {request.max_tokens}")
        
        # Generate response using Fireworks API
        response_data = generate_chat_response_with_api(
            request.model_id,
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty
        )
        
        return InferenceResponse(
            generated_text=response_data["generated_text"],
            model_id=request.model_id,
            generation_time=response_data["generation_time"],
            tokens_generated=response_data["tokens_generated"],
            finish_reason=response_data["finish_reason"]
        )
        
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

@app.post("/contract/review")
async def contract_review(request: ContractReviewRequest):
    """Specialized endpoint for contract review using Fireworks API"""
    try:
        logger.info(f"Contract review request received:")
        logger.info(f"  - Model ID: {request.model_id}")
        logger.info(f"  - Review type: {request.review_type}")
        logger.info(f"  - Number of sections: {len(request.contract_sections)}")
        
        # Generate real contract review using the Fireworks API
        logger.info("Calling real contract review generation...")
        review_text = generate_contract_review_with_api(request.contract_sections, request.model_id)
        
        return {
            "review": review_text,
            "model_id": request.model_id,
            "review_type": request.review_type,
            "sections_reviewed": len(request.contract_sections),
            "generation_time": 0,  # This is handled by the API
            "tokens_generated": len(review_text.split())
        }
        
    except Exception as e:
        logger.error(f"Error in contract review: {e}")
        raise HTTPException(status_code=500, detail=f"Contract review failed: {str(e)}")

@app.post("/batch/completions")
async def batch_completions(request: BatchInferenceRequest):
    """Process multiple chat completion requests in batch"""
    results = []
    
    # Process requests with concurrency limit
    semaphore = asyncio.Semaphore(request.max_concurrent)
    
    async def process_single_request(req: InferenceRequest):
        async with semaphore:
            try:
                response = await chat_completions(req)
                return {"success": True, "response": response}
            except Exception as e:
                return {"success": False, "error": str(e), "request_id": req.model_id}
    
    # Execute all requests
    tasks = [process_single_request(req) for req in request.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        "results": results,
        "total_requests": len(request.requests),
        "successful": len([r for r in results if isinstance(r, dict) and r.get("success")])
    }

@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    models = fireworks_client.find_fireworks_models()
    model_info = next((m for m in models if m["id"] == model_id), None)
    
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model_id,
        "status": "available",
        "type": model_info["type"],
        "model_info": model_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9200) 