#!/usr/bin/env python3
"""
Training Client for Enhanced Model Training Service

This client provides an easy interface to interact with the enhanced training service,
including job submission, monitoring, and result retrieval.
"""

import requests
import json
import time
import uuid
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import sys

class TrainingClient:
    """Client for interacting with the enhanced model training service"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the training service is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available models"""
        response = self.session.get(f"{self.base_url}/models/available")
        response.raise_for_status()
        return response.json()["models"]
    
    def get_strategies(self) -> Dict:
        """Get available training strategies and LoRA sizes"""
        response = self.session.get(f"{self.base_url}/strategies")
        response.raise_for_status()
        return response.json()
    
    def get_hardware_info(self) -> Dict:
        """Get hardware information"""
        response = self.session.get(f"{self.base_url}/hardware")
        response.raise_for_status()
        return response.json()
    
    def analyze_dataset(self, dataset_files: List[str], dataset_format: str = "auto") -> Dict:
        """Analyze dataset without training"""
        data = {
            "dataset_files": dataset_files,
            "dataset_format": dataset_format
        }
        response = self.session.post(f"{self.base_url}/dataset/analyze", json=data)
        response.raise_for_status()
        return response.json()
    
    def start_training(self, 
                      model_name: str,
                      dataset_files: List[str],
                      job_id: Optional[str] = None,
                      dataset_format: str = "auto",
                      training_strategy: str = "balanced",
                      lora_size: str = "medium",
                      use_lora: bool = True,
                      learning_rate: Optional[float] = None,
                      num_epochs: Optional[int] = None,
                      batch_size: Optional[int] = None,
                      max_length: Optional[int] = None,
                      validation_split: float = 0.1,
                      use_wandb: bool = False,
                      early_stopping: bool = True,
                      save_steps: Optional[int] = None) -> Dict:
        """Start a training job"""
        
        if job_id is None:
            job_id = f"train_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        data = {
            "job_id": job_id,
            "model_name": model_name,
            "dataset_files": dataset_files,
            "dataset_format": dataset_format,
            "training_strategy": training_strategy,
            "lora_size": lora_size,
            "use_lora": use_lora,
            "validation_split": validation_split,
            "use_wandb": use_wandb,
            "early_stopping": early_stopping
        }
        
        # Add optional parameters
        if learning_rate is not None:
            data["learning_rate"] = learning_rate
        if num_epochs is not None:
            data["num_epochs"] = num_epochs
        if batch_size is not None:
            data["batch_size"] = batch_size
        if max_length is not None:
            data["max_length"] = max_length
        if save_steps is not None:
            data["save_steps"] = save_steps
        
        response = self.session.post(f"{self.base_url}/train", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a specific training job"""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all training jobs"""
        response = self.session.get(f"{self.base_url}/jobs")
        response.raise_for_status()
        return response.json()["jobs"]
    
    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a training job"""
        response = self.session.delete(f"{self.base_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def monitor_job(self, job_id: str, poll_interval: int = 30) -> Dict:
        """Monitor a training job until completion"""
        print(f"Monitoring job {job_id}...")
        
        while True:
            try:
                status = self.get_job_status(job_id)
                
                # Print status update
                print(f"\rStatus: {status['status']} | "
                      f"Epoch: {status['current_epoch']}/{status['total_epochs']} | "
                      f"Step: {status['current_step']}/{status['total_steps']} | "
                      f"Progress: {status['progress']:.1f}%", end="")
                
                if status.get('loss'):
                    print(f" | Loss: {status['loss']:.4f}", end="")
                
                # Check if job is complete
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    print(f"\n\nJob {job_id} finished with status: {status['status']}")
                    if status['status'] == 'completed':
                        print(f"Model saved to: {status.get('model_path', 'Unknown')}")
                        if status.get('final_loss'):
                            print(f"Final loss: {status['final_loss']:.4f}")
                        if status.get('train_runtime'):
                            print(f"Training time: {status['train_runtime']:.2f} seconds")
                    elif status['status'] == 'failed':
                        print(f"Error: {status.get('error', 'Unknown error')}")
                    return status
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                print(f"\n\nMonitoring interrupted. Job {job_id} is still running.")
                return self.get_job_status(job_id)
            except Exception as e:
                print(f"\nError monitoring job: {e}")
                time.sleep(poll_interval)

def print_models(client: TrainingClient):
    """Print available models"""
    try:
        models = client.get_available_models()
        print("\n🤖 Available Models:")
        print("-" * 80)
        for model in models:
            print(f"ID: {model['id']}")
            print(f"Name: {model['name']}")
            print(f"Parameters: {model['parameters']}")
            print(f"Max Length: {model['max_length']}")
            print(f"Recommended Batch Size: {model['recommended_batch_size']}")
            print(f"Description: {model['description']}")
            print("-" * 80)
    except Exception as e:
        print(f"Error getting models: {e}")

def print_strategies(client: TrainingClient):
    """Print available training strategies"""
    try:
        strategies = client.get_strategies()
        print("\n🎯 Training Strategies:")
        for strategy in strategies['strategies']:
            print(f"  - {strategy}")
        
        print("\n🔧 LoRA Sizes:")
        for size in strategies['lora_sizes']:
            print(f"  - {size}")
    except Exception as e:
        print(f"Error getting strategies: {e}")

def print_hardware(client: TrainingClient):
    """Print hardware information"""
    try:
        hardware = client.get_hardware_info()
        print("\n💻 Hardware Information:")
        print(f"Platform: {hardware['platform']}")
        print(f"Machine: {hardware['machine']}")
        print(f"Device: {hardware['device']}")
        print(f"Hardware Type: {hardware['hardware_type']}")
        if hardware.get('device_count', 1) > 1:
            print(f"Device Count: {hardware['device_count']}")
        if hardware.get('memory_gb'):
            print(f"Memory: {hardware['memory_gb']:.1f} GB")
    except Exception as e:
        print(f"Error getting hardware info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Model Training Client")
    parser.add_argument("--url", default="http://localhost:8002", help="Training service URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check
    subparsers.add_parser("health", help="Check service health")
    
    # List models
    subparsers.add_parser("models", help="List available models")
    
    # List strategies
    subparsers.add_parser("strategies", help="List training strategies")
    
    # Hardware info
    subparsers.add_parser("hardware", help="Show hardware information")
    
    # Analyze dataset
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset")
    analyze_parser.add_argument("files", nargs="+", help="Dataset files")
    analyze_parser.add_argument("--format", default="auto", help="Dataset format")
    
    # Start training
    train_parser = subparsers.add_parser("train", help="Start training job")
    train_parser.add_argument("model", help="Model name")
    train_parser.add_argument("files", nargs="+", help="Dataset files")
    train_parser.add_argument("--job-id", help="Custom job ID")
    train_parser.add_argument("--format", default="auto", help="Dataset format")
    train_parser.add_argument("--strategy", default="balanced", help="Training strategy")
    train_parser.add_argument("--lora-size", default="medium", help="LoRA size")
    train_parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--max-length", type=int, help="Max sequence length")
    train_parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    train_parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    train_parser.add_argument("--save-steps", type=int, help="Save checkpoint every N steps")
    train_parser.add_argument("--monitor", action="store_true", help="Monitor job until completion")
    
    # Job status
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="Job ID")
    
    # List jobs
    subparsers.add_parser("jobs", help="List all jobs")
    
    # Monitor job
    monitor_parser = subparsers.add_parser("monitor", help="Monitor job")
    monitor_parser.add_argument("job_id", help="Job ID")
    monitor_parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    
    # Cancel job
    cancel_parser = subparsers.add_parser("cancel", help="Cancel job")
    cancel_parser.add_argument("job_id", help="Job ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    client = TrainingClient(args.url)
    
    try:
        if args.command == "health":
            health = client.health_check()
            print(f"Service Status: {health['status']}")
            if health['status'] == 'healthy':
                print(f"Ray Initialized: {health.get('ray_initialized', False)}")
                print(f"Active Jobs: {health.get('active_jobs', 0)}")
                if 'hardware' in health:
                    print(f"Hardware: {health['hardware']['hardware_type']}")
            else:
                print(f"Error: {health.get('error', 'Unknown')}")
        
        elif args.command == "models":
            print_models(client)
        
        elif args.command == "strategies":
            print_strategies(client)
        
        elif args.command == "hardware":
            print_hardware(client)
        
        elif args.command == "analyze":
            print(f"Analyzing dataset: {args.files}")
            result = client.analyze_dataset(args.files, args.format)
            print(f"\n📊 Dataset Analysis:")
            print(f"Total Examples: {result['total_examples']}")
            print(f"Average Length: {result['avg_length']:.1f} tokens")
            print(f"Format Distribution: {result['format_distribution']}")
            print(f"Length Distribution: {result['length_distribution']}")
        
        elif args.command == "train":
            print(f"Starting training job...")
            print(f"Model: {args.model}")
            print(f"Dataset: {args.files}")
            print(f"Strategy: {args.strategy}")
            print(f"LoRA: {'Disabled' if args.no_lora else f'Enabled ({args.lora_size})'}")
            
            result = client.start_training(
                model_name=args.model,
                dataset_files=args.files,
                job_id=args.job_id,
                dataset_format=args.format,
                training_strategy=args.strategy,
                lora_size=args.lora_size,
                use_lora=not args.no_lora,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
                validation_split=args.validation_split,
                use_wandb=args.wandb,
                early_stopping=not args.no_early_stopping,
                save_steps=args.save_steps
            )
            
            print(f"\n✅ Training job started!")
            print(f"Job ID: {result['job_id']}")
            print(f"Status: {result['status']}")
            
            if args.monitor:
                client.monitor_job(result['job_id'])
        
        elif args.command == "status":
            status = client.get_job_status(args.job_id)
            print(f"\n📋 Job Status: {args.job_id}")
            print(f"Status: {status['status']}")
            print(f"Progress: {status['progress']:.1f}%")
            print(f"Epoch: {status['current_epoch']}/{status['total_epochs']}")
            print(f"Step: {status['current_step']}/{status['total_steps']}")
            if status.get('loss'):
                print(f"Loss: {status['loss']:.4f}")
            if status.get('model_path'):
                print(f"Model Path: {status['model_path']}")
        
        elif args.command == "jobs":
            jobs = client.get_all_jobs()
            print(f"\n📋 All Training Jobs ({len(jobs)} total):")
            print("-" * 100)
            for job in jobs:
                print(f"ID: {job['job_id']}")
                print(f"Model: {job.get('model_name', 'Unknown')}")
                print(f"Status: {job['status']}")
                print(f"Progress: {job['progress']:.1f}%")
                if job.get('started_at'):
                    print(f"Started: {job['started_at']}")
                print("-" * 100)
        
        elif args.command == "monitor":
            client.monitor_job(args.job_id, args.interval)
        
        elif args.command == "cancel":
            result = client.cancel_job(args.job_id)
            print(f"Job {args.job_id} cancelled: {result['status']}")
    
    except requests.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the training service is running.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 