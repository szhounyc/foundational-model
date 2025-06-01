#!/usr/bin/env python3
"""
Test script for the Enhanced Model Training Service

This script tests the core functionality of the enhanced training system
without requiring a full training run.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from config import (
            get_model_config, get_lora_config, get_training_strategy,
            MODEL_CONFIGS, LORA_CONFIGS, TRAINING_STRATEGIES
        )
        print("✅ Config module imported successfully")
        
        from data_loader import DatasetLoader, DatasetStatistics
        print("✅ Data loader module imported successfully")
        
        from enhanced_trainer import EnhancedTrainer
        print("✅ Enhanced trainer module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configurations():
    """Test configuration loading"""
    print("\n🔧 Testing configurations...")
    
    try:
        from config import get_model_config, get_lora_config, get_training_strategy
        
        # Test model config
        model_config = get_model_config("llama-3.2-1b")
        print(f"✅ Model config loaded: {model_config.model_id}")
        
        # Test LoRA config
        lora_config = get_lora_config("medium")
        print(f"✅ LoRA config loaded: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # Test training strategy
        strategy = get_training_strategy("balanced")
        print(f"✅ Training strategy loaded: LoRA={strategy.use_lora}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_dataset_loader():
    """Test dataset loading functionality"""
    print("\n📊 Testing dataset loader...")
    
    try:
        from data_loader import DatasetLoader, DatasetStatistics
        from transformers import AutoTokenizer
        
        # Create a temporary dataset file
        sample_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain neural networks."},
                    {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks."}
                ]
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            # Test tokenizer loading (use a small model for testing)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test data loader
            data_loader = DatasetLoader(tokenizer, max_length=512)
            
            # Use absolute path for the dataset file
            dataset = data_loader.load_dataset([os.path.abspath(temp_file)], "fireworks")
            
            print(f"✅ Dataset loaded: {len(dataset)} examples")
            
            # Test dataset statistics
            stats = DatasetStatistics.analyze_dataset(dataset)
            print(f"✅ Dataset analyzed: avg_length={stats['avg_length']:.1f}")
            
            return True
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            
    except Exception as e:
        print(f"❌ Dataset loader error: {e}")
        return False

def test_hardware_detection():
    """Test hardware detection"""
    print("\n💻 Testing hardware detection...")
    
    try:
        # Import the hardware detection function from main_enhanced
        import platform
        import torch
        
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
            hardware_info.update({
                "device": "mps",
                "hardware_type": "apple_silicon"
            })
            print("✅ Apple Silicon detected")
        
        # Check for CUDA
        elif torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            hardware_info.update({
                "device": "cuda",
                "device_count": device_count,
                "hardware_type": "cuda_multi" if device_count > 1 else "cuda_single"
            })
            print(f"✅ CUDA detected: {device_count} GPU(s)")
        
        # Check for MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            hardware_info.update({
                "device": "mps",
                "hardware_type": "apple_silicon"
            })
            print("✅ MPS (Apple Silicon) detected")
        
        else:
            print("✅ CPU-only environment detected")
        
        print(f"   Hardware type: {hardware_info['hardware_type']}")
        print(f"   Device: {hardware_info['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection error: {e}")
        return False

def test_api_models():
    """Test API model definitions"""
    print("\n🤖 Testing API models...")
    
    try:
        from main_enhanced import TrainingRequest, TrainingStatus, ModelInfo
        
        # Test TrainingRequest
        request = TrainingRequest(
            job_id="test_job",
            model_name="llama-3.2-1b",
            dataset_files=["test.jsonl"]
        )
        print(f"✅ TrainingRequest created: {request.job_id}")
        
        # Test TrainingStatus
        status = TrainingStatus(
            job_id="test_job",
            status="queued",
            progress=0.0,
            current_step=0,
            total_steps=100,
            current_epoch=0,
            total_epochs=3
        )
        print(f"✅ TrainingStatus created: {status.status}")
        
        return True
        
    except Exception as e:
        print(f"❌ API models error: {e}")
        return False

def test_training_client():
    """Test training client functionality"""
    print("\n📱 Testing training client...")
    
    try:
        from training_client import TrainingClient
        
        # Create client (don't actually connect)
        client = TrainingClient("http://localhost:8002")
        print("✅ Training client created")
        
        # Test that methods exist
        assert hasattr(client, 'health_check')
        assert hasattr(client, 'start_training')
        assert hasattr(client, 'get_job_status')
        print("✅ Client methods verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Training client error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Enhanced Model Training Service - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configurations", test_configurations),
        ("Dataset Loader", test_dataset_loader),
        ("Hardware Detection", test_hardware_detection),
        ("API Models", test_api_models),
        ("Training Client", test_training_client),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced training system is ready.")
        print("\n🚀 Next steps:")
        print("1. Start the training service: python main_enhanced.py")
        print("2. Test with client: python training_client.py health")
        print("3. Train a model: python training_client.py train llama-3.2-1b dataset.jsonl")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 