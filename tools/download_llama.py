#!/usr/bin/env python3
"""
Download Llama 3.2 1B model with authentication and cache locally
"""

import os
import requests
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, HfApi

def setup_hf_auth():
    """Set up Hugging Face authentication"""
    try:
        # Try to get token from environment or HF cache
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Please run: huggingface-cli login")
        return False

def download_llama_with_auth():
    """Download Llama 3.2 1B model with authentication"""
    
    if not setup_hf_auth():
        return False, None
    
    # Try the official Llama 3.2 1B model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    try:
        print(f"🦙 Downloading {model_name} with authentication...")
        
        # Download tokenizer with authentication
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=True,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer downloaded for {model_name}")
        
        # Download model with authentication
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=True,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Successfully downloaded {model_name}")
        print(f"📁 Model cached in ~/.cache/huggingface/hub/")
        
        return True, model_name
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        
        # If the main model fails, try the base version
        try:
            base_model = "meta-llama/Llama-3.2-1B"
            print(f"🦙 Trying base model {base_model}...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                use_auth_token=True,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                use_auth_token=True,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Successfully downloaded {base_model}")
            return True, base_model
            
        except Exception as e2:
            print(f"❌ Failed to download base model: {e2}")
            return False, None

def try_alternative_small_models():
    """Try other small language models as alternatives"""
    
    print("🔄 Trying alternative small language models...")
    
    alternative_models = [
        "microsoft/DialoGPT-small",
        "gpt2",
        "distilgpt2",
        "microsoft/DialoGPT-medium",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B"
    ]
    
    for model_name in alternative_models:
        try:
            print(f"📥 Trying to download {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            print(f"✅ Successfully downloaded {model_name}")
            print(f"📁 Model cached in ~/.cache/huggingface/hub/")
            
            return True, model_name
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            continue
    
    return False, None

def download_llama_model():
    """Main download function - prioritize authenticated Llama models"""
    
    print("🚀 Starting Llama 3.2 1B model download with authentication...")
    
    # First try to get actual Llama models with auth
    success, model_name = download_llama_with_auth()
    if success:
        print(f"🎉 Successfully downloaded Llama model: {model_name}")
        return True
    
    print("⚠️ Could not download Llama models, trying alternatives...")
    
    # If Llama fails, try alternatives
    success, model_name = try_alternative_small_models()
    if success:
        print(f"🎉 Successfully downloaded alternative model: {model_name}")
        print("💡 You may need to update the inference service to use this model")
        return True
    
    print("❌ All download attempts failed")
    return False

if __name__ == "__main__":
    success = download_llama_model()
    if success:
        print("🎉 Model successfully cached!")
        print("🔧 You can now restart the inference service")
    else:
        print("💥 Failed to download any model")
        print("💡 You may need to:")
        print("   1. Request access to Llama models at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("   2. Run: huggingface-cli login")
        print("   3. Check your internet connection") 