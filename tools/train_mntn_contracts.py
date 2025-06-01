#!/usr/bin/env python3
"""
Training script for MNTN contracts dataset using the enhanced trainer
"""

import json
import requests
import time
from pathlib import Path

def start_training_job():
    """Start a training job for the MNTN contracts dataset"""
    
    # Training configuration
    training_config = {
        "model_name": "microsoft/DialoGPT-small",  # Start with a smaller model for testing
        "dataset_path": "mntn_fireworks_dataset/mntn_contracts_fireworks.jsonl",
        "output_dir": "mntn_contract_model",
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "save_steps": 50,
        "eval_steps": 50,
        "logging_steps": 10,
        "max_length": 512,
        "use_lora": True,
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "lora_dropout": 0.1
        },
        "training_strategy": "lora",
        "hardware_type": "auto"
    }
    
    print("🚀 Starting MNTN Contracts Training Job")
    print("=" * 60)
    print(f"Model: {training_config['model_name']}")
    print(f"Dataset: {training_config['dataset_path']}")
    print(f"Output: {training_config['output_dir']}")
    print(f"Strategy: {training_config['training_strategy']}")
    print(f"LoRA: {training_config['use_lora']}")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = Path(training_config['dataset_path'])
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return None
    
    print(f"✅ Dataset found: {dataset_path} ({dataset_path.stat().st_size / 1024:.1f} KB)")
    
    # Start training via API
    try:
        response = requests.post(
            "http://localhost:8000/train",
            json=training_config,
            timeout=30
        )
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data.get('job_id')
            print(f"✅ Training job started successfully!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {job_data.get('status')}")
            return job_id
        else:
            print(f"❌ Failed to start training job: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to training service. Is it running?")
        print("   Start it with: cd model-trainer && python main.py")
        return None
    except Exception as e:
        print(f"❌ Error starting training job: {e}")
        return None

def monitor_training_job(job_id):
    """Monitor the training job progress"""
    
    print(f"\n📊 Monitoring Training Job: {job_id}")
    print("=" * 60)
    
    while True:
        try:
            response = requests.get(f"http://localhost:8000/jobs/{job_id}")
            
            if response.status_code == 200:
                job_status = response.json()
                status = job_status.get('status')
                
                print(f"Status: {status}")
                
                if 'progress' in job_status:
                    progress = job_status['progress']
                    print(f"Progress: {progress}")
                
                if 'current_step' in job_status:
                    print(f"Current Step: {job_status['current_step']}")
                
                if 'loss' in job_status:
                    print(f"Current Loss: {job_status['loss']:.4f}")
                
                if status in ['completed', 'failed', 'cancelled']:
                    print(f"\n🎯 Training job {status}!")
                    if status == 'completed':
                        print("✅ Model training completed successfully!")
                        if 'model_path' in job_status:
                            print(f"   Model saved to: {job_status['model_path']}")
                    break
                
                print("-" * 40)
                time.sleep(30)  # Check every 30 seconds
                
            else:
                print(f"❌ Error checking job status: {response.status_code}")
                break
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
            break

def main():
    """Main training workflow"""
    
    print("🏗️  MNTN Contracts Model Training")
    print("=" * 60)
    
    # Start training job
    job_id = start_training_job()
    
    if job_id:
        # Monitor training progress
        monitor_training_job(job_id)
    else:
        print("\n❌ Failed to start training job")
        print("\nTroubleshooting:")
        print("1. Make sure the training service is running:")
        print("   cd model-trainer && python main.py")
        print("2. Check that the dataset file exists:")
        print("   ls -la mntn_fireworks_dataset/")
        print("3. Verify the dataset format:")
        print("   python validate_fireworks_dataset.py")

if __name__ == "__main__":
    main() 