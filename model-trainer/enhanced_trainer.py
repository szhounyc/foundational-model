#!/usr/bin/env python3
"""
Enhanced Model Trainer with LoRA Support

This module provides an enhanced trainer with support for LoRA fine-tuning,
better monitoring, evaluation, and hardware optimization.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, DatasetDict
import numpy as np
from datetime import datetime
import wandb

# LoRA imports
try:
    from peft import (
        LoraConfig, get_peft_model, TaskType,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. LoRA training will be disabled.")

# Quantization imports
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    logging.warning("BitsAndBytesConfig not available. Quantization will be disabled.")

from config import ModelConfig, LoRAConfig, TrainingStrategy
from data_loader import DatasetLoader, DatasetStatistics

logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """Enhanced trainer with LoRA support and advanced monitoring"""
    
    def __init__(
        self,
        job_id: str,
        model_config: ModelConfig,
        lora_config: Optional[LoRAConfig] = None,
        training_strategy: Optional[TrainingStrategy] = None,
        hardware_type: str = "cpu"
    ):
        self.job_id = job_id
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_strategy = training_strategy or TrainingStrategy()
        self.hardware_type = hardware_type
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_stats = {
            "start_time": None,
            "end_time": None,
            "total_steps": 0,
            "current_step": 0,
            "best_loss": float('inf'),
            "training_loss_history": [],
            "eval_loss_history": [],
            "learning_rate_history": []
        }
        
        # Setup output directory
        self.output_dir = Path(f"/app/checkpoints/{job_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if available
        self.use_wandb = self._init_wandb()
    
    def _init_wandb(self) -> bool:
        """Initialize Weights & Biases logging if available"""
        try:
            wandb.init(
                project="llm-fine-tuning",
                name=f"job-{self.job_id}",
                config={
                    "model": self.model_config.model_id,
                    "job_id": self.job_id,
                    "hardware": self.hardware_type,
                    "use_lora": self.training_strategy.use_lora,
                    **self.model_config.__dict__
                },
                mode="disabled" if os.getenv("WANDB_DISABLED") else "online"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            return False
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with optimizations"""
        logger.info(f"Loading model: {self.model_config.model_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_id,
            trust_remote_code=True
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if requested
        quantization_config = None
        if QUANTIZATION_AVAILABLE and (self.training_strategy.use_4bit or self.training_strategy.use_8bit):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.training_strategy.use_4bit,
                load_in_8bit=self.training_strategy.use_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.hardware_type == "cuda" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif self.hardware_type == "cuda":
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            **model_kwargs
        )
        
        # Apply LoRA if requested
        if self.training_strategy.use_lora and PEFT_AVAILABLE:
            self._apply_lora()
        
        # Move to appropriate device
        if self.hardware_type == "mps":
            self.model = self.model.to("mps")
        elif self.hardware_type == "cuda" and not quantization_config:
            self.model = self.model.to("cuda")
        
        logger.info(f"Model loaded successfully. Parameters: {self._count_parameters()}")
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model"""
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not available, skipping LoRA")
            return
        
        logger.info("Applying LoRA configuration...")
        
        # Prepare model for k-bit training if using quantization
        if self.training_strategy.use_4bit or self.training_strategy.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=self.lora_config.target_modules,
            bias=self.lora_config.bias
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params
        }
    
    def prepare_training_args(self, dataset_size: int) -> TrainingArguments:
        """Prepare training arguments"""
        # Calculate steps
        steps_per_epoch = dataset_size // (
            self.model_config.batch_size * self.model_config.gradient_accumulation_steps
        )
        
        if self.model_config.max_steps:
            max_steps = self.model_config.max_steps
            num_train_epochs = max_steps / steps_per_epoch
        else:
            num_train_epochs = self.model_config.num_train_epochs
            max_steps = int(steps_per_epoch * num_train_epochs)
        
        self.training_stats["total_steps"] = max_steps
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            gradient_accumulation_steps=self.model_config.gradient_accumulation_steps,
            learning_rate=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
            warmup_steps=self.model_config.warmup_steps,
            
            # Precision and optimization
            fp16=self.model_config.fp16 and self.hardware_type == "cuda",
            bf16=self.model_config.bf16 and self.hardware_type == "cuda",
            gradient_checkpointing=self.model_config.gradient_checkpointing,
            
            # Logging and saving
            logging_steps=self.model_config.logging_steps,
            save_steps=self.model_config.save_steps,
            eval_steps=self.model_config.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=self.model_config.load_best_model_at_end,
            metric_for_best_model=self.model_config.metric_for_best_model,
            greater_is_better=self.model_config.greater_is_better,
            
            # Data loading
            dataloader_num_workers=self.model_config.dataloader_num_workers,
            dataloader_pin_memory=self.hardware_type == "cuda",
            
            # Evaluation
            evaluation_strategy="steps" if self.model_config.eval_steps > 0 else "no",
            
            # Reporting
            report_to=["wandb"] if self.use_wandb else [],
            run_name=f"job-{self.job_id}",
            
            # Hardware specific
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
        )
        
        return training_args
    
    def train(self, dataset: DatasetDict) -> Dict[str, Any]:
        """Train the model"""
        logger.info("Starting training...")
        self.training_stats["start_time"] = datetime.now()
        
        # Prepare training arguments
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("validation")
        
        training_args = self.prepare_training_args(len(train_dataset))
        
        # Create data collator
        data_loader = DatasetLoader(self.tokenizer, self.model_config.max_length)
        data_collator = data_loader.get_data_collator()
        
        # Create custom trainer
        self.trainer = CustomTrainer(
            job_id=self.job_id,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else [],
            training_stats=self.training_stats
        )
        
        # Start training
        try:
            train_result = self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Save training statistics
            self.training_stats["end_time"] = datetime.now()
            self._save_training_stats(train_result)
            
            logger.info("Training completed successfully!")
            return {
                "status": "completed",
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "output_dir": str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _save_training_stats(self, train_result):
        """Save detailed training statistics"""
        stats_file = self.output_dir / "training_stats.json"
        
        stats = {
            "job_id": self.job_id,
            "model_config": self.model_config.__dict__,
            "training_strategy": self.training_strategy.__dict__,
            "hardware_type": self.hardware_type,
            "training_stats": self.training_stats,
            "final_metrics": train_result.metrics,
            "model_parameters": self._count_parameters()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Training statistics saved to {stats_file}")

class CustomTrainer(Trainer):
    """Custom trainer with enhanced logging and monitoring"""
    
    def __init__(self, job_id: str, training_stats: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id
        self.training_stats = training_stats
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with statistics tracking"""
        super().log(logs)
        
        # Update training statistics
        if "loss" in logs:
            self.training_stats["training_loss_history"].append({
                "step": self.state.global_step,
                "loss": logs["loss"],
                "timestamp": datetime.now().isoformat()
            })
            
            if logs["loss"] < self.training_stats["best_loss"]:
                self.training_stats["best_loss"] = logs["loss"]
        
        if "eval_loss" in logs:
            self.training_stats["eval_loss_history"].append({
                "step": self.state.global_step,
                "eval_loss": logs["eval_loss"],
                "timestamp": datetime.now().isoformat()
            })
        
        if "learning_rate" in logs:
            self.training_stats["learning_rate_history"].append({
                "step": self.state.global_step,
                "learning_rate": logs["learning_rate"],
                "timestamp": datetime.now().isoformat()
            })
        
        self.training_stats["current_step"] = self.state.global_step
        
        # Log progress
        if self.state.global_step % 50 == 0:
            progress = (self.state.global_step / self.training_stats["total_steps"]) * 100
            logger.info(f"Job {self.job_id}: Step {self.state.global_step}/{self.training_stats['total_steps']} ({progress:.1f}%)")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with monitoring"""
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def evaluate_model(model_path: str, test_dataset: Dataset, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """Evaluate a trained model on a test dataset"""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for example in test_dataset:
            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item()
            total_samples += 1
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": total_samples
    } 