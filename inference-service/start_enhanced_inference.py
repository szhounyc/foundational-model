#!/usr/bin/env python3
"""
Enhanced Inference Service Startup Script

This script starts the enhanced inference service with proper configuration
for Fireworks models and contract review functionality.
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    print("🔧 Loading environment variables...")
    
    # Try to load .env from parent directory first
    env_path = Path("../.env")
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        # Try current directory
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded environment from {env_path}")
        else:
            print("⚠️ No .env file found, using system environment variables")
    
    # Check for required API key
    fireworks_key = os.getenv("FIREWORKS_API_KEY")
    if fireworks_key:
        print(f"✅ FIREWORKS_API_KEY found: {fireworks_key[:10]}...")
        return True
    else:
        print("❌ FIREWORKS_API_KEY not found in environment")
        print("Please ensure your .env file contains FIREWORKS_API_KEY=your_api_key")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "torch", "transformers", 
        "peft", "accelerate", "safetensors"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        # Install from requirements.txt
        requirements_path = Path("requirements.txt")
        if requirements_path.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_path)
            ], check=True)
        else:
            # Install individual packages
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages, check=True)
        
        print("✅ Dependencies installed")
    else:
        print("✅ All dependencies satisfied")

def check_models():
    """Check if Fireworks models are available"""
    print("\n🔍 Checking for Fireworks models...")
    
    # Check models directory
    models_dir = Path("/app/models")
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("Please ensure the models directory is properly mounted")
        return False
    
    # Look for Fireworks model structure
    fireworks_models = []
    for model_path in models_dir.rglob("adapter_config.json"):
        if "checkpoint" in str(model_path):
            fireworks_models.append(model_path.parent)
    
    if fireworks_models:
        print(f"✅ Found {len(fireworks_models)} Fireworks model(s):")
        for model in fireworks_models:
            print(f"   📁 {model}")
        return True
    else:
        print("❌ No Fireworks models found")
        print("   Expected structure: models/*/checkpoint/adapter_config.json")
        return False

def check_datasets():
    """Check if processed datasets are available"""
    print("\n🔍 Checking for processed datasets...")
    
    dataset_dir = Path("../feature-engineering/fireworks/processed_datasets")
    if not dataset_dir.exists():
        print("❌ Processed datasets directory not found")
        return False
    
    dataset_files = list(dataset_dir.glob("*.jsonl"))
    if dataset_files:
        print(f"✅ Found {len(dataset_files)} dataset file(s):")
        for dataset in dataset_files:
            print(f"   📄 {dataset.name}")
        return True
    else:
        print("❌ No dataset files found")
        return False

def start_service():
    """Start the enhanced inference service"""
    print("\n🚀 Starting Enhanced Inference Service...")
    
    # We're already in the inference-service directory
    service_dir = Path(".")
    
    # Start the service
    cmd = [
        sys.executable, "-m", "uvicorn",
        "enhanced_inference:app",
        "--host", "0.0.0.0",
        "--port", "9200",
        "--reload"
    ]
    
    print(f"📡 Starting service on http://localhost:9200")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        # Start the service in the background
        process = subprocess.Popen(
            cmd,
            cwd=service_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if service is running
        if process.poll() is None:
            print("✅ Service started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Service failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        return None

def wait_for_service(max_wait=30):
    """Wait for the service to be ready"""
    print(f"\n⏳ Waiting for service to be ready (max {max_wait}s)...")
    
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:9200/health", timeout=2)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except:
            pass
        
        print(f"   Waiting... ({i+1}/{max_wait})")
        time.sleep(1)
    
    print("❌ Service did not become ready in time")
    return False

def run_quick_test():
    """Run a quick test to verify the service is working"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:9200/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Hardware: {health_data.get('hardware', {}).get('device', 'unknown')}")
            
            # Test model discovery
            response = requests.get("http://localhost:9200/models", timeout=30)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])
                print(f"✅ Model discovery: Found {len(models)} model(s)")
                
                if models:
                    for model in models:
                        print(f"   📦 {model['id']} ({model.get('type', 'unknown')})")
                
                return True
            else:
                print(f"❌ Model discovery failed: {response.status_code}")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main startup sequence"""
    print("🎯 Enhanced Inference Service Startup")
    print("=" * 50)
    
    # Step 0: Load environment variables
    if not load_environment():
        print("\n❌ Failed to load required environment variables")
        return 1
    
    # Step 1: Check dependencies
    try:
        check_dependencies()
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return 1
    
    # Step 2: Check models
    if not check_models():
        print("\n⚠️ Warning: No Fireworks models found")
        print("   The service will start but model loading may fail")
    
    # Step 3: Check datasets
    if not check_datasets():
        print("\n⚠️ Warning: No processed datasets found")
        print("   Testing with real data may not work")
    
    # Step 4: Start service
    process = start_service()
    if not process:
        print("\n❌ Failed to start service")
        return 1
    
    # Step 5: Wait for service to be ready
    if not wait_for_service():
        print("\n❌ Service not ready")
        process.terminate()
        return 1
    
    # Step 6: Run quick test
    if not run_quick_test():
        print("\n❌ Quick test failed")
        process.terminate()
        return 1
    
    # Success!
    print("\n" + "=" * 50)
    print("🎉 Enhanced Inference Service is ready!")
    print("=" * 50)
    print("📡 Service URL: http://localhost:9200")
    print("📚 API Docs: http://localhost:9200/docs")
    print("🧪 Run tests: python test_enhanced_inference.py")
    print("\n💡 Available endpoints:")
    print("   GET  /health                    - Service health")
    print("   GET  /models                    - List available models")
    print("   POST /models/{id}/load          - Load a model")
    print("   POST /chat/completions          - Chat completions")
    print("   POST /contract/review           - Contract review")
    print("   POST /batch/completions         - Batch processing")
    
    print(f"\n🔄 Service running (PID: {process.pid})")
    print("   Press Ctrl+C to stop")
    
    try:
        # Keep the service running
        process.wait()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping service...")
        process.terminate()
        process.wait()
        print("✅ Service stopped")
    
    return 0

if __name__ == "__main__":
    exit(main()) 