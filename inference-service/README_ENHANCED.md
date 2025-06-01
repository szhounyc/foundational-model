# Enhanced Inference Service

A powerful inference service designed specifically for Fireworks fine-tuned models with specialized contract review capabilities.

## 🎯 Features

- **Fireworks Model Support**: Native support for LoRA adapters from Fireworks.ai
- **Contract Review**: Specialized endpoints for legal contract analysis
- **Chat Completions**: OpenAI-compatible chat completion API
- **Batch Processing**: Efficient batch inference for multiple requests
- **Hardware Detection**: Automatic GPU/CPU detection and optimization
- **Model Management**: Dynamic model loading and unloading
- **Real-time Monitoring**: Health checks and performance metrics

## 🏗️ Architecture

```
Enhanced Inference Service
├── Model Management
│   ├── Fireworks LoRA Adapter Loading
│   ├── Base Model Integration (Qwen2.5-7B-Instruct)
│   └── Hardware Optimization
├── API Endpoints
│   ├── Chat Completions (/chat/completions)
│   ├── Contract Review (/contract/review)
│   ├── Batch Processing (/batch/completions)
│   └── Model Management (/models/*)
└── Testing Framework
    ├── Comprehensive Test Suite
    ├── Real Data Testing
    └── Performance Benchmarks
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU inference)
- 10GB+ disk space for models

### Dependencies
```
fastapi>=0.104.0
uvicorn>=0.24.0
torch>=2.1.0
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
safetensors>=0.4.0
pydantic>=2.5.0
numpy>=1.24.0
requests>=2.31.0
python-multipart>=0.0.6
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd inference-service
pip install -r requirements.txt
```

### 2. Start the Service
```bash
# Option A: Use the startup script (recommended)
python ../start_enhanced_inference.py

# Option B: Start manually
uvicorn enhanced_inference:app --host 0.0.0.0 --port 8003 --reload
```

### 3. Verify Installation
```bash
# Check service health
curl http://localhost:8003/health

# List available models
curl http://localhost:8003/models
```

### 4. Run Tests
```bash
python ../test_enhanced_inference.py
```

## 📁 Model Structure

The service expects Fireworks models in this structure:
```
models/
└── {job_id}/
    └── {checkpoint_id}/
        └── {model_name}/
            └── checkpoint/
                ├── adapter_config.json
                ├── adapter_model.safetensors
                ├── train_config.json
                └── stats.json
```

Example:
```
models/sftj-s1xkr35z/552ca5/zlg-re-fm-sl-mntn/checkpoint/
```

## 🔌 API Reference

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "hardware": {
    "device": "cuda",
    "gpu_count": 1,
    "gpu_memory": "24GB"
  }
}
```

### List Models
```http
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "id": "zlg-re-fm-sl-mntn",
      "type": "fireworks_lora",
      "base_model": "qwen2p5-7b-instruct",
      "status": "available",
      "path": "models/sftj-s1xkr35z/552ca5/zlg-re-fm-sl-mntn"
    }
  ]
}
```

### Load Model
```http
POST /models/{model_id}/load
```

**Response:**
```json
{
  "status": "loaded",
  "model_id": "zlg-re-fm-sl-mntn",
  "load_time": 15.2,
  "memory_usage": "8.5GB"
}
```

### Chat Completions
```http
POST /chat/completions
Content-Type: application/json

{
  "model_id": "zlg-re-fm-sl-mntn",
  "messages": [
    {
      "role": "system",
      "content": "You are a legal contract review expert."
    },
    {
      "role": "user",
      "content": "Please review this contract clause: 'The buyer shall close within 30 days of the effective date.'"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.3
}
```

**Response:**
```json
{
  "generated_text": "This clause establishes a clear timeline for closing...",
  "generation_time": 2.1,
  "token_count": 156,
  "model_id": "zlg-re-fm-sl-mntn"
}
```

### Contract Review
```http
POST /contract/review
Content-Type: application/json

{
  "contract_sections": [
    "Section 5.1: The closing date shall be no earlier than December 1, 2024.",
    "Section 7.1: Any notice under this Agreement shall be in writing."
  ],
  "model_id": "zlg-re-fm-sl-mntn",
  "review_type": "comprehensive"
}
```

**Response:**
```json
{
  "review": "Comprehensive review of the provided sections...",
  "sections_reviewed": 2,
  "issues_identified": 3,
  "recommendations": [
    "Consider adding a specific time for closing",
    "Clarify delivery method for notices"
  ],
  "generation_time": 3.5
}
```

### Batch Processing
```http
POST /batch/completions
Content-Type: application/json

{
  "requests": [
    {
      "model_id": "zlg-re-fm-sl-mntn",
      "messages": [...],
      "max_tokens": 256
    },
    {
      "model_id": "zlg-re-fm-sl-mntn",
      "messages": [...],
      "max_tokens": 256
    }
  ],
  "max_concurrent": 2
}
```

**Response:**
```json
{
  "results": [
    {
      "success": true,
      "generated_text": "Response to first request...",
      "generation_time": 1.8
    },
    {
      "success": true,
      "generated_text": "Response to second request...",
      "generation_time": 2.1
    }
  ],
  "total_requests": 2,
  "successful": 2,
  "failed": 0,
  "total_time": 4.2
}
```

## 🧪 Testing

### Comprehensive Test Suite

The service includes a comprehensive test suite that validates:

1. **Service Health**: Basic connectivity and status
2. **Model Discovery**: Finding available Fireworks models
3. **Model Loading**: Loading LoRA adapters correctly
4. **Chat Completions**: Basic text generation
5. **Contract Review**: Specialized contract analysis
6. **Real Data Testing**: Using actual contract datasets
7. **Batch Processing**: Multiple concurrent requests
8. **Performance**: Response times and quality

### Running Tests

```bash
# Run all tests
python test_enhanced_inference.py

# Check test results
cat inference_test_results.json
```

### Test Data

Tests use real contract data from:
- `feature-engineering/fireworks/processed_datasets/mntn_contracts_training.jsonl`
- Manual test cases for edge cases

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
MODELS_DIR=./models
BASE_MODEL_NAME=qwen2p5-7b-instruct

# Service configuration
HOST=0.0.0.0
PORT=8003
MAX_CONCURRENT_REQUESTS=10

# Hardware configuration
DEVICE=auto  # auto, cuda, cpu
MAX_MEMORY_GB=16
```

### Model Configuration

Models are automatically discovered based on directory structure. Configuration is read from:
- `adapter_config.json`: LoRA adapter settings
- `train_config.json`: Training configuration
- `stats.json`: Training statistics

## 📊 Performance

### Benchmarks

| Model | Hardware | Tokens/sec | Memory Usage | Load Time |
|-------|----------|------------|--------------|-----------|
| MNTN Contract Model | RTX 4090 | 45 | 8.5GB | 15s |
| MNTN Contract Model | CPU | 8 | 4.2GB | 25s |

### Optimization Tips

1. **GPU Memory**: Use `torch.cuda.empty_cache()` between requests
2. **Batch Size**: Adjust based on available memory
3. **Model Precision**: Use FP16 for faster inference
4. **Concurrent Requests**: Limit based on memory constraints

## 🐛 Troubleshooting

### Common Issues

**Model Not Found**
```
Error: No Fireworks models found
Solution: Check model directory structure and adapter_config.json
```

**CUDA Out of Memory**
```
Error: CUDA out of memory
Solution: Reduce batch size or use CPU inference
```

**Import Errors**
```
Error: No module named 'peft'
Solution: Install requirements: pip install -r requirements.txt
```

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Monitor service health:
```bash
# Check if service is running
curl http://localhost:8003/health

# Check model status
curl http://localhost:8003/models/{model_id}/info

# Check system resources
nvidia-smi  # For GPU usage
htop        # For CPU/memory usage
```

## 🔒 Security

### API Security

- Input validation for all endpoints
- Request size limits
- Rate limiting (recommended for production)
- Model access controls

### Model Security

- Secure model loading from trusted paths
- Validation of model configurations
- Memory isolation between requests

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8003

CMD ["uvicorn", "enhanced_inference:app", "--host", "0.0.0.0", "--port", "8003"]
```

### Load Balancing

For high-traffic scenarios:
- Use multiple service instances
- Implement load balancing (nginx, HAProxy)
- Consider model sharding for large models

### Monitoring

Recommended monitoring:
- Response times and error rates
- Model memory usage
- GPU utilization
- Request queue length

## 📝 License

This enhanced inference service is part of the foundational model project and follows the same licensing terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review test results for diagnostics
3. Check logs for detailed error messages
4. Create an issue with reproduction steps

---

**Last Updated**: January 2024  
**Version**: 1.0.0  
**Compatibility**: Fireworks.ai LoRA models, Qwen2.5-7B-Instruct base model 