#!/usr/bin/env python3
"""
Data Loading and Preprocessing for Model Training

This module handles loading and preprocessing of different dataset formats
for fine-tuning, including Fireworks.ai chat format, Alpaca format, and plain text.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import torch
import os

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handles loading and preprocessing of training datasets"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has required tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Chat template for conversation formatting
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = self._get_default_chat_template()
    
    def _get_default_chat_template(self) -> str:
        """Get default chat template for models without one"""
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'user' %}"
            "### Human: {{ message['content'] }}\n\n"
            "{% elif message['role'] == 'assistant' %}"
            "### Assistant: {{ message['content'] }}\n\n"
            "{% endif %}"
            "{% endfor %}"
        )
    
    def load_dataset(self, dataset_files: List[str], dataset_format: str = "auto") -> Dataset:
        """Load dataset from files"""
        logger.info(f"Loading dataset from {len(dataset_files)} files...")
        all_examples = []
        
        for file_path in dataset_files:
            # Handle both absolute and relative paths
            if os.path.isabs(file_path):
                full_path = Path(file_path)
            else:
                # Try relative to current directory first
                full_path = Path(file_path)
                if not full_path.exists():
                    # Try relative to /app/data for Docker compatibility
                    full_path = Path(f"/app/data/{file_path}")
            
            if not full_path.exists():
                logger.warning(f"Dataset file not found: {full_path}")
                continue
            
            try:
                examples = self._load_file(full_path, dataset_format)
                all_examples.extend(examples)
                logger.info(f"Loaded {len(examples)} examples from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_examples:
            raise ValueError("No training data could be loaded")
        
        logger.info(f"Total examples loaded: {len(all_examples)}")
        return Dataset.from_list(all_examples)
    
    def _load_file(self, file_path: Path, dataset_format: str) -> List[Dict]:
        """Load examples from a single file"""
        examples = []
        
        if file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if data:  # Skip empty lines
                            examples.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                elif isinstance(data, dict):
                    if 'chunks' in data:
                        # Handle processed PDF format
                        examples = [{"text": chunk["text"]} for chunk in data['chunks']]
                    else:
                        examples = [data]
        
        # Auto-detect format if not specified
        if dataset_format == "auto":
            dataset_format = self._detect_format(examples)
        
        # Convert to standardized format
        return self._convert_format(examples, dataset_format)
    
    def _detect_format(self, examples: List[Dict]) -> str:
        """Automatically detect dataset format"""
        if not examples:
            return "text"
        
        first_example = examples[0]
        
        # Check for Fireworks.ai chat format
        if "messages" in first_example:
            return "fireworks"
        
        # Check for Alpaca format
        if all(key in first_example for key in ["instruction", "output"]):
            return "alpaca"
        
        # Check for plain text
        if "text" in first_example:
            return "text"
        
        # Default to text format
        return "text"
    
    def _convert_format(self, examples: List[Dict], dataset_format: str) -> List[Dict]:
        """Convert examples to standardized format"""
        converted = []
        
        for example in examples:
            try:
                if dataset_format == "fireworks":
                    converted_example = self._convert_fireworks_format(example)
                elif dataset_format == "alpaca":
                    converted_example = self._convert_alpaca_format(example)
                elif dataset_format == "text":
                    converted_example = self._convert_text_format(example)
                else:
                    logger.warning(f"Unknown format: {dataset_format}, treating as text")
                    converted_example = self._convert_text_format(example)
                
                if converted_example:
                    converted.append(converted_example)
            except Exception as e:
                logger.warning(f"Error converting example: {e}")
                continue
        
        return converted
    
    def _convert_fireworks_format(self, example: Dict) -> Optional[Dict]:
        """Convert Fireworks.ai chat format to training format"""
        if "messages" not in example:
            return None
        
        messages = example["messages"]
        if not messages:
            return None
        
        try:
            # Use tokenizer's chat template to format the conversation
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            return {
                "text": formatted_text,
                "messages": messages,
                "format": "fireworks"
            }
        except Exception as e:
            logger.warning(f"Error applying chat template: {e}")
            # Fallback to manual formatting
            return self._manual_chat_format(messages)
    
    def _manual_chat_format(self, messages: List[Dict]) -> Dict:
        """Manually format chat messages when template fails"""
        formatted_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        return {
            "text": "\n\n".join(formatted_parts),
            "messages": messages,
            "format": "fireworks"
        }
    
    def _convert_alpaca_format(self, example: Dict) -> Optional[Dict]:
        """Convert Alpaca format to training format"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if not instruction or not output:
            return None
        
        # Format as instruction-following conversation
        if input_text:
            formatted_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return {
            "text": formatted_text,
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "format": "alpaca"
        }
    
    def _convert_text_format(self, example: Dict) -> Optional[Dict]:
        """Convert plain text format to training format"""
        text = example.get("text", "")
        if not text or len(text.strip()) < 10:
            return None
        
        return {
            "text": text.strip(),
            "format": "text"
        }
    
    def prepare_dataset(self, dataset: Dataset, validation_split: float = 0.1) -> DatasetDict:
        """Prepare dataset for training with tokenization and train/val split"""
        logger.info("Tokenizing dataset...")
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize texts
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=False,
            )
            
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Split into train and validation
        if validation_split > 0:
            split_dataset = tokenized_dataset.train_test_split(
                test_size=validation_split,
                seed=42
            )
            return DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })
        else:
            return DatasetDict({
                "train": tokenized_dataset
            })
    
    def get_data_collator(self):
        """Get data collator for dynamic padding"""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
            pad_to_multiple_of=8,  # For efficiency on modern hardware
        )

class DatasetStatistics:
    """Calculate and display dataset statistics"""
    
    @staticmethod
    def analyze_dataset(dataset: Dataset) -> Dict:
        """Analyze dataset and return statistics"""
        stats = {
            "total_examples": len(dataset),
            "avg_length": 0,
            "min_length": float('inf'),
            "max_length": 0,
            "format_distribution": {},
            "length_distribution": {
                "0-100": 0,
                "100-500": 0,
                "500-1000": 0,
                "1000-2000": 0,
                "2000+": 0
            }
        }
        
        total_length = 0
        
        for example in dataset:
            text = example.get("text", "")
            length = len(text)
            
            total_length += length
            stats["min_length"] = min(stats["min_length"], length)
            stats["max_length"] = max(stats["max_length"], length)
            
            # Format distribution
            format_type = example.get("format", "unknown")
            stats["format_distribution"][format_type] = stats["format_distribution"].get(format_type, 0) + 1
            
            # Length distribution
            if length <= 100:
                stats["length_distribution"]["0-100"] += 1
            elif length <= 500:
                stats["length_distribution"]["100-500"] += 1
            elif length <= 1000:
                stats["length_distribution"]["500-1000"] += 1
            elif length <= 2000:
                stats["length_distribution"]["1000-2000"] += 1
            else:
                stats["length_distribution"]["2000+"] += 1
        
        stats["avg_length"] = total_length / len(dataset) if len(dataset) > 0 else 0
        
        return stats
    
    @staticmethod
    def print_statistics(stats: Dict):
        """Print formatted dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total Examples: {stats['total_examples']:,}")
        print(f"Average Length: {stats['avg_length']:.1f} characters")
        print(f"Min Length: {stats['min_length']:,} characters")
        print(f"Max Length: {stats['max_length']:,} characters")
        
        print("\nFormat Distribution:")
        for format_type, count in stats['format_distribution'].items():
            percentage = (count / stats['total_examples']) * 100
            print(f"  {format_type}: {count:,} ({percentage:.1f}%)")
        
        print("\nLength Distribution:")
        for length_range, count in stats['length_distribution'].items():
            percentage = (count / stats['total_examples']) * 100
            print(f"  {length_range} chars: {count:,} ({percentage:.1f}%)")
        print("="*50 + "\n") 