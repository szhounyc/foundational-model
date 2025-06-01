# Enhanced Model Training Service

A comprehensive fine-tuning service for Large Language Models with LoRA support, hardware optimization, and advanced monitoring capabilities.

## 🚀 Features

- **LoRA Fine-tuning**: Parameter-efficient training with configurable LoRA sizes
- **Hardware Optimization**: Automatic detection and optimization for Apple Silicon (M4 Max), CUDA, and CPU
- **Multiple Model Support**: Support for Llama, DeepSeek, Mistral, Phi-3, and Qwen models
- **Dataset Formats**: Auto-detection for Fireworks.ai, Alpaca, and plain text formats
- **Advanced Monitoring**: Real-time training progress with optional Weights & Biases integration
- **Distributed Training**: Ray-based distributed training support
- **RESTful API**: Complete API for job management and monitoring

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- FastAPI
- Ray (for distributed training)

## 🛠️ Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Apple Silicon optimization:
```bash
# MPS acceleration is automatically detected
```

3. For CUDA support:
```bash
# CUDA will be automatically detected if available
```

## 🏃‍♂️ Quick Start

### 1. Start the Training Service

```bash
# Using the enhanced main file
python main_enhanced.py

# Or using uvicorn directly
uvicorn main_enhanced:app --host 0.0.0.0 --port 8002
```

### 2. Using the Training Client

```bash
# Check service health
python training_client.py health

# List available models
python training_client.py models

# Analyze your dataset
python training_client.py analyze path/to/dataset.jsonl

# Start training
python training_client.py train llama-3.2-1b path/to/dataset.jsonl --monitor
```

## 🤖 Supported Models

| Model | Size | Description |
|-------|------|-------------|
| `llama-3.2-1b` | 1.2B | Fast and efficient for most tasks |
| `llama-3.2-3b` | 3.2B | Balanced performance and speed |
| `llama-3.1-8b` | 8.0B | High performance for complex tasks |
| `deepseek-coder-1.3b` | 1.3B | Optimized for code generation |
| `deepseek-coder-6.7b` | 6.7B | Advanced code understanding |
| `mistral-7b` | 7.2B | Excellent general-purpose model |
| `phi-3-mini` | 3.8B | Compact and efficient |
| `qwen2.5-1.5b` | 1.5B | Fast multilingual model |
| `qwen2.5-7b` | 7.6B | Advanced multilingual capabilities |

## 🎯 Training Strategies

### Available Strategies

- **`fast`**: Quick training with minimal resources
- **`balanced`**: Balanced performance and quality (default)
- **`quality`**: High-quality training with more resources
- **`memory_efficient`**: Optimized for limited memory
- **`apple_optimized`**: Optimized for Apple Silicon

### LoRA Configurations

- **`small`**: r=8, alpha=16 - Minimal parameter overhead
- **`medium`**: r=16, alpha=32 - Balanced efficiency (default)
- **`large`**: r=32, alpha=64 - Higher capacity
- **`xlarge`**: r=64, alpha=128 - Maximum capacity

## 📊 Dataset Formats

### Fireworks.ai Format (Recommended)
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Alpaca Format
```json
{
  "instruction": "Explain machine learning",
  "input": "",
  "output": "Machine learning is..."
}
```

### Plain Text Format
```
Human: What is machine learning?
Assistant: Machine learning is...
```

## 🔧 API Reference

### Start Training Job

```bash
POST /train
```

**Request Body:**
```json
{
  "job_id": "my_training_job",
  "model_name": "llama-3.2-1b",
  "dataset_files": ["path/to/dataset.jsonl"],
  "dataset_format": "fireworks",
  "training_strategy": "balanced",
  "lora_size": "medium",
  "use_lora": true,
  "learning_rate": 2e-4,
  "num_epochs": 3,
  "batch_size": 4,
  "validation_split": 0.1,
  "use_wandb": false
}
```

### Monitor Job Status

```bash
GET /jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "my_training_job",
  "status": "running",
  "progress": 45.2,
  "current_step": 150,
  "total_steps": 332,
  "current_epoch": 2,
  "total_epochs": 3,
  "loss": 0.8234,
  "learning_rate": 1.8e-4
}
```

### Other Endpoints

- `GET /health` - Service health check
- `GET /models/available` - List available models
- `GET /strategies` - List training strategies
- `GET /hardware` - Hardware information
- `GET /jobs` - List all jobs
- `DELETE /jobs/{job_id}` - Cancel job
- `POST /dataset/analyze` - Analyze dataset

## 💻 Hardware Optimization

### Apple Silicon (M4 Max)
- Automatic MPS acceleration
- Optimized batch sizes and memory usage
- Special configurations for M4 Max chips

### CUDA GPUs
- Multi-GPU support with Ray
- Automatic memory optimization
- Mixed precision training

### CPU Fallback
- Optimized for CPU-only environments
- Reduced batch sizes and memory usage

## 📈 Training Examples

### Basic Training
```bash
python training_client.py train llama-3.2-1b dataset.jsonl
```

### Advanced Training with Custom Parameters
```bash
python training_client.py train llama-3.2-3b dataset.jsonl \
  --strategy quality \
  --lora-size large \
  --learning-rate 1e-4 \
  --epochs 5 \
  --batch-size 2 \
  --wandb \
  --monitor
```

### Memory-Efficient Training
```bash
python training_client.py train llama-3.1-8b dataset.jsonl \
  --strategy memory_efficient \
  --lora-size small \
  --batch-size 1
```

### Code-Specific Training
```bash
python training_client.py train deepseek-coder-1.3b code_dataset.jsonl \
  --strategy balanced \
  --max-length 4096
```

## 🔍 Monitoring and Logging

### Real-time Monitoring
```bash
# Monitor a specific job
python training_client.py monitor job_12345

# Check job status
python training_client.py status job_12345

# List all jobs
python training_client.py jobs
```

### Weights & Biases Integration
```bash
python training_client.py train llama-3.2-1b dataset.jsonl --wandb
```

### Training Logs
- Automatic logging to `checkpoints/{job_id}/training_stats.json`
- Real-time loss tracking
- Hardware utilization metrics

## 🐳 Docker Support

### Build and Run
```bash
# Build the container
docker build -t enhanced-model-trainer .

# Run the service
docker run -p 8002:8002 enhanced-model-trainer
```

### Docker Compose
```yaml
version: '3.8'
services:
  model-trainer:
    build: .
    ports:
      - "8002:8002"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./datasets:/app/datasets
```

## 🔧 Configuration

### Model Configuration (`config.py`)
```python
# Customize model settings
MODEL_CONFIGS["custom-model"] = ModelConfig(
    model_id="path/to/model",
    max_length=2048,
    batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3
)
```

### Training Strategy
```python
# Add custom training strategy
TRAINING_STRATEGIES["custom"] = TrainingStrategy(
    use_lora=True,
    gradient_checkpointing=True,
    fp16=True,
    dataloader_num_workers=2
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size: `--batch-size 1`
   - Use memory-efficient strategy: `--strategy memory_efficient`
   - Use smaller LoRA: `--lora-size small`

2. **Slow Training**
   - Check hardware detection: `python training_client.py hardware`
   - Increase batch size if memory allows
   - Use fast strategy: `--strategy fast`

3. **Model Loading Issues**
   - Ensure model name is correct: `python training_client.py models`
   - Check internet connection for model downloads
   - Verify disk space for model cache

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=.
python -m pdb main_enhanced.py
```

## 📝 Examples

### Training a Contract Review Model
```bash
# Using the processed Fireworks.ai dataset
python training_client.py train llama-3.2-3b \
  ../feature-engineering/fireworks/processed_datasets/mntn_contracts_training.jsonl \
  --job-id contract_review_v1 \
  --strategy balanced \
  --lora-size medium \
  --epochs 3 \
  --validation-split 0.15 \
  --monitor
```

### Code Generation Fine-tuning
```bash
python training_client.py train deepseek-coder-1.3b code_dataset.jsonl \
  --strategy quality \
  --max-length 4096 \
  --learning-rate 1e-4 \
  --epochs 2
```

### Multilingual Model Training
```bash
python training_client.py train qwen2.5-7b multilingual_dataset.jsonl \
  --strategy apple_optimized \
  --lora-size large \
  --batch-size 2
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers team
- Microsoft for LoRA implementation
- Ray team for distributed training
- FastAPI for the web framework 