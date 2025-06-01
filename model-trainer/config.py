#!/usr/bin/env python3
"""
Model Training Configuration

This module contains configurations for different models, training strategies,
and hardware optimizations for the fine-tuning pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_id: str
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: Optional[int] = None
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning"""
    r: int = 16  # Rank
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False

@dataclass
class TrainingStrategy:
    """Training strategy configuration"""
    use_lora: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    use_8bit: bool = False
    use_4bit: bool = False

# Model configurations optimized for different hardware setups
MODEL_CONFIGS = {
    "llama-3.2-1b": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B",
        max_length=2048,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=100,
        num_train_epochs=3,
        fp16=True,
        gradient_checkpointing=True
    ),
    "llama-3.2-3b": ModelConfig(
        model_id="meta-llama/Llama-3.2-3B",
        max_length=2048,
        batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=3,
        fp16=True,
        gradient_checkpointing=True
    ),
    "llama-3.1-8b": ModelConfig(
        model_id="meta-llama/Meta-Llama-3.1-8B",
        max_length=2048,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=200,
        num_train_epochs=2,
        fp16=True,
        gradient_checkpointing=True
    ),
    "deepseek-coder-1.3b": ModelConfig(
        model_id="deepseek-ai/deepseek-coder-1.3b-base",
        max_length=2048,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=100,
        num_train_epochs=3,
        fp16=True,
        gradient_checkpointing=True
    ),
    "deepseek-coder-6.7b": ModelConfig(
        model_id="deepseek-ai/deepseek-coder-6.7b-base",
        max_length=2048,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=200,
        num_train_epochs=2,
        fp16=True,
        gradient_checkpointing=True
    ),
    "mistral-7b": ModelConfig(
        model_id="mistralai/Mistral-7B-v0.1",
        max_length=2048,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=200,
        num_train_epochs=2,
        fp16=True,
        gradient_checkpointing=True
    ),
    "phi-3-mini": ModelConfig(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        max_length=2048,
        batch_size=6,
        gradient_accumulation_steps=3,
        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=3,
        fp16=True,
        gradient_checkpointing=True
    ),
    "qwen2.5-1.5b": ModelConfig(
        model_id="Qwen/Qwen2.5-1.5B",
        max_length=2048,
        batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=100,
        num_train_epochs=3,
        fp16=True,
        gradient_checkpointing=True
    ),
    "qwen2.5-7b": ModelConfig(
        model_id="Qwen/Qwen2.5-7B",
        max_length=2048,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=200,
        num_train_epochs=2,
        fp16=True,
        gradient_checkpointing=True
    )
}

# LoRA configurations for different model sizes
LORA_CONFIGS = {
    "small": LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1
    ),
    "medium": LoRAConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1
    ),
    "large": LoRAConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05
    )
}

# Training strategies for different scenarios
TRAINING_STRATEGIES = {
    "fast": TrainingStrategy(
        use_lora=True,
        use_gradient_checkpointing=True,
        use_flash_attention=False,
        use_4bit=True
    ),
    "balanced": TrainingStrategy(
        use_lora=True,
        use_gradient_checkpointing=True,
        use_flash_attention=True,
        use_8bit=False
    ),
    "quality": TrainingStrategy(
        use_lora=False,
        use_gradient_checkpointing=True,
        use_flash_attention=True,
        use_8bit=False
    ),
    "memory_efficient": TrainingStrategy(
        use_lora=True,
        use_gradient_checkpointing=True,
        use_flash_attention=True,
        use_4bit=True
    )
}

# Hardware-specific optimizations
HARDWARE_CONFIGS = {
    "apple_m4_max": {
        "device": "mps",
        "fp16": False,  # MPS doesn't support fp16 well
        "bf16": False,
        "dataloader_num_workers": 8,
        "pin_memory": False
    },
    "apple_silicon": {
        "device": "mps",
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": 4,
        "pin_memory": False
    },
    "cuda_single": {
        "device": "cuda",
        "fp16": True,
        "bf16": False,
        "dataloader_num_workers": 4,
        "pin_memory": True
    },
    "cuda_multi": {
        "device": "cuda",
        "fp16": True,
        "bf16": True,
        "dataloader_num_workers": 8,
        "pin_memory": True,
        "ddp": True
    },
    "cpu": {
        "device": "cpu",
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": 2,
        "pin_memory": False
    }
}

# Dataset format configurations
DATASET_FORMATS = {
    "fireworks": {
        "chat_template": True,
        "system_role": "system",
        "user_role": "user",
        "assistant_role": "assistant",
        "message_key": "messages"
    },
    "alpaca": {
        "instruction_key": "instruction",
        "input_key": "input",
        "output_key": "output"
    },
    "text": {
        "text_key": "text"
    }
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def get_lora_config(size: str = "medium") -> LoRAConfig:
    """Get LoRA configuration by size"""
    if size not in LORA_CONFIGS:
        raise ValueError(f"Unsupported LoRA size: {size}. Available sizes: {list(LORA_CONFIGS.keys())}")
    return LORA_CONFIGS[size]

def get_training_strategy(strategy: str = "balanced") -> TrainingStrategy:
    """Get training strategy by name"""
    if strategy not in TRAINING_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy}. Available strategies: {list(TRAINING_STRATEGIES.keys())}")
    return TRAINING_STRATEGIES[strategy]

def get_hardware_config(hardware_type: str) -> Dict:
    """Get hardware configuration by type"""
    if hardware_type not in HARDWARE_CONFIGS:
        raise ValueError(f"Unsupported hardware: {hardware_type}. Available types: {list(HARDWARE_CONFIGS.keys())}")
    return HARDWARE_CONFIGS[hardware_type]

def adjust_config_for_hardware(model_config: ModelConfig, hardware_type: str) -> ModelConfig:
    """Adjust model configuration based on hardware capabilities"""
    hardware_config = get_hardware_config(hardware_type)
    
    # Adjust batch size and other parameters based on hardware
    if hardware_type == "apple_m4_max":
        # M4 Max has excellent memory bandwidth
        model_config.batch_size = min(model_config.batch_size * 2, 16)
        model_config.dataloader_num_workers = 8
        model_config.fp16 = False  # MPS works better with fp32
    elif hardware_type == "apple_silicon":
        model_config.dataloader_num_workers = 4
        model_config.fp16 = False
    elif hardware_type == "cuda_multi":
        # Multi-GPU setup can handle larger batches
        model_config.batch_size = min(model_config.batch_size * 2, 32)
        model_config.dataloader_num_workers = 8
    elif hardware_type == "cpu":
        # CPU training needs smaller batches
        model_config.batch_size = max(model_config.batch_size // 2, 1)
        model_config.gradient_accumulation_steps *= 2
        model_config.fp16 = False
        model_config.dataloader_num_workers = 2
    
    return model_config 