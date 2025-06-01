# Contract Review Platform

A comprehensive AI-powered contract analysis platform with fine-tuned language models, web interface, and cloud-hosted inference through Fireworks.ai.

## 🏗️ Project Architecture

```
foundational-model/
├── 📁 frontend/                    # React Web Interface
│   ├── src/                        # React source code
│   ├── public/                     # Static assets & favicon
│   ├── package.json               # Frontend dependencies
│   └── Dockerfile                 # Frontend container
│
├── 📁 backend/                     # Backend API Services
│   ├── src/                       # Backend source code
│   ├── package.json              # Backend dependencies
│   └── Dockerfile                # Backend container
│
├── 📁 inference-service/          # AI Inference Service
│   ├── enhanced_inference.py      # Main inference service (Fireworks.ai)
│   ├── start_enhanced_inference.py # Service startup script
│   ├── test_enhanced_inference.py # Comprehensive test suite
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                # Inference container
│
├── 📁 model-trainer/              # Model Training & Management
│   ├── enhanced_trainer.py        # Training with LoRA support
│   ├── train_mntn_contracts.py    # Contract-specific training
│   ├── training_client.py         # Training API client
│   └── requirements.txt           # Training dependencies
│
├── 📁 feature-engineering/        # Data Processing Pipeline
│   └── fireworks/                 # Fireworks.ai data processing
│       ├── dataset_processor_fireworks.py
│       ├── validate_fireworks_dataset.py
│       └── processed_datasets/    # Training data
│
├── 📁 models/                     # Fine-tuned Model Artifacts
│   └── sftj-*/                   # Fireworks model deployments
│       └── checkpoint/            # LoRA adapters & configs
│
├── 📁 dataset/                    # Raw Contract Data
│   └── zlg-re/MNTN/              # MNTN legal contracts
│
├── docker-compose.yml             # Full stack orchestration
├── .env                          # Environment configuration
└── README.md                     # This documentation
```

## 🌟 Key Features

### 🔍 **AI-Powered Contract Analysis**
- **Specialized Models**: Fine-tuned Llama 3.2 models for legal contract review
- **Cloud Inference**: Hosted on Fireworks.ai for scalable, high-performance inference
- **Multiple Review Types**: Quick analysis, detailed review, and custom workflows

### 🖥️ **Modern Web Interface**
- **React Frontend**: Clean, responsive UI with Material-UI components
- **Real-time Analysis**: Live contract review with streaming responses
- **Review History**: Track and manage previous contract analyses
- **Professional Design**: Custom favicon and branded interface

### ⚡ **High-Performance Infrastructure**
- **Microservices Architecture**: Containerized frontend, backend, and inference services
- **Docker Orchestration**: Complete deployment with docker-compose
- **API Gateway**: RESTful APIs with proper error handling and logging
- **Scalable Deployment**: Production-ready with health checks and monitoring

## 🚀 Quick Start

### 1. **Prerequisites**

```bash
# Required software
- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

# Required accounts & API keys
- Fireworks.ai account and API key
- Hugging Face account (for model access)
```

### 2. **Environment Setup**

```bash
# Clone the repository
git clone <repository-url>
cd foundational-model

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
```

**Required Environment Variables:**
```bash
# Fireworks.ai Configuration
FIREWORKS_API_KEY=fw_your_api_key_here

# Hugging Face Configuration  
HUGGINGFACE_API_KEY=hf_your_token_here

# Service Configuration
FRONTEND_PORT=9000
BACKEND_PORT=9100
INFERENCE_PORT=9200
```

### 3. **One-Command Deployment**

```bash
# Start the complete platform
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

**Services will be available at:**
- 🌐 **Frontend**: http://localhost:9000 (Contract Review UI)
- 🔧 **Backend**: http://localhost:9100 (API Services)
- 🤖 **Inference**: http://localhost:9200 (AI Models)

### 4. **Verify Deployment**

```bash
# Test inference service health
curl http://localhost:9200/health

# Test contract review API
curl -X POST http://localhost:9200/contract/review \
  -H "Content-Type: application/json" \
  -d '{
    "contract_sections": ["This lease agreement establishes terms between landlord and tenant for rental property."],
    "model_id": "sftj-qbplmzw9",
    "review_type": "quick"
  }'

# Open web interface
open http://localhost:9000
```

## 🎯 Using the Platform

### **Web Interface Workflow**

1. **Navigate to Dashboard** (`http://localhost:9000`)
   - View platform overview and recent activity
   - Access quick stats and system health

2. **Contract Review** (`/contract-review`)
   - Upload contract documents or paste text
   - Select review type (Quick, Detailed, Custom)
   - Choose fine-tuned model for analysis
   - Get AI-powered insights and recommendations

3. **Review History** (`/review-history`)
   - Browse previous contract analyses
   - Export results and manage data
   - Track analysis patterns over time

### **API Integration**

```python
# Python example for direct API usage
import requests

# Contract review request
response = requests.post('http://localhost:9200/contract/review', json={
    "contract_sections": [
        "Tenant agrees to pay rent of $2000 monthly.",
        "Lease term is 12 months starting January 1, 2024."
    ],
    "model_id": "sftj-qbplmzw9",
    "review_type": "detailed"
})

analysis = response.json()
print(f"Review Score: {analysis['overall_score']}")
print(f"Issues Found: {len(analysis['issues'])}")
```

## 🔌 API Reference

### **Inference Service** (Port 9200)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health and model status |
| `/models` | GET | List available fine-tuned models |
| `/contract/review` | POST | Analyze contract sections |
| `/chat/completions` | POST | OpenAI-compatible chat API |

### **Backend Service** (Port 9100)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/contracts` | GET/POST | Contract management |
| `/api/reviews` | GET/POST | Review history |
| `/api/models` | GET | Available AI models |
| `/api/health` | GET | Backend service status |

## 🤖 AI Models & Training

### **Available Models**

| Model ID | Base Model | Specialization | Performance |
|----------|------------|----------------|-------------|
| `sftj-qbplmzw9` | Llama 3.2 1B | General contracts | Fast, efficient |
| `sftj-s1xkr35z` | Llama 3.2 1B | Real estate | High accuracy |
| `sftj-wst3swj0` | Llama 3.2 1B | Commercial | Detailed analysis |

### **Training Pipeline**

```bash
# Process new contract data
cd feature-engineering/fireworks
python dataset_processor_fireworks.py --input ../dataset/new_contracts/

# Validate training data
python validate_fireworks_dataset.py

# Train new model (via Fireworks.ai)
cd model-trainer
python train_mntn_contracts.py --dataset mntn_contracts_training.jsonl
```

### **Custom Model Integration**

1. **Train on Fireworks.ai**: Upload data and train via Fireworks console
2. **Deploy Model**: Get deployment ID from Fireworks
3. **Update Configuration**: Add model ID to inference service
4. **Test Integration**: Verify model works with contract review API

## 🐳 Deployment Options

### **Development Environment**

```bash
# Run services locally
cd frontend && npm start     # Port 3000
cd backend && npm start      # Port 9100
cd inference-service && python enhanced_inference.py  # Port 9200
```

### **Production Deployment**

```bash
# Build optimized containers
docker compose -f docker-compose.yml build --no-cache

# Deploy with resource limits
docker compose up -d --scale inference-service=2

# Enable monitoring
docker compose logs -f | grep -E "(ERROR|WARN)"
```

### **Cloud Deployment (AWS/GCP/Azure)**

```bash
# Push to container registry
docker tag contract-frontend:latest your-registry/contract-frontend:v1.0.0
docker push your-registry/contract-frontend:v1.0.0

# Deploy with Kubernetes/ECS/Cloud Run
# (See deployment guides for specific platforms)
```

## 🔧 Configuration

### **Fireworks.ai Setup**

1. **Create Account**: Sign up at https://fireworks.ai
2. **Get API Key**: Generate API key in dashboard
3. **Configure Models**: Note your deployed model IDs
4. **Set Environment**: Add `FIREWORKS_API_KEY` to `.env`

### **Model Configuration**

```json
{
  "model_mappings": {
    "sftj-qbplmzw9": {
      "deployment": "accounts/your-account/deployedModels/your-deployment",
      "description": "General contract analysis",
      "max_tokens": 2048
    }
  }
}
```

### **Performance Tuning**

```bash
# Adjust inference settings
INFERENCE_TIMEOUT=120        # API timeout in seconds
MAX_CONCURRENT_REQUESTS=10   # Concurrent request limit
BATCH_SIZE=4                # Batch processing size

# Frontend optimization
REACT_APP_API_BASE_URL=http://localhost:9100
GENERATE_SOURCEMAP=false     # Faster builds
```

## 🧪 Testing & Quality Assurance

### **Automated Testing**

```bash
# Run all tests
docker compose -f docker-compose.test.yml up

# Test specific components
cd inference-service && python test_enhanced_inference.py
cd frontend && npm test
cd backend && npm test
```

### **Manual Testing Scenarios**

1. **Contract Upload**: Test various document formats
2. **API Performance**: Load test with multiple concurrent requests
3. **Error Handling**: Test with invalid inputs and network issues
4. **Browser Compatibility**: Test across Chrome, Firefox, Safari

### **Model Validation**

```bash
# Validate model outputs
cd inference-service
python test_enhanced_inference.py --test-suite contract_analysis

# Performance benchmarking
python benchmark_models.py --models sftj-qbplmzw9,sftj-s1xkr35z
```

## 🐛 Troubleshooting

### **Common Issues**

**🔸 Fireworks API Timeouts**
```bash
# Solution: Increase timeout or check API status
curl -H "Authorization: Bearer $FIREWORKS_API_KEY" \
     https://api.fireworks.ai/inference/v1/models
```

**🔸 Container Memory Issues**
```bash
# Solution: Increase Docker memory limits
docker system prune -a
docker compose up -d --memory 4g
```

**🔸 Frontend Build Failures**
```bash
# Solution: Clear cache and rebuild
cd frontend && rm -rf node_modules package-lock.json
npm install && npm run build
```

**🔸 Model Loading Errors**
```bash
# Check model directory structure
ls -la models/sftj-*/checkpoint/
# Should contain: adapter_config.json, adapter_model.safetensors

# Verify environment variables
docker compose config | grep FIREWORKS_API_KEY
```

### **Debugging Commands**

```bash
# View detailed logs
docker compose logs -f inference-service | grep ERROR

# Check service health
curl http://localhost:9200/health | jq .

# Monitor resource usage
docker stats contract-inference-service

# Debug API requests
curl -v -X POST http://localhost:9200/contract/review \
  -H "Content-Type: application/json" \
  -d '{"contract_sections":["test"],"model_id":"sftj-qbplmzw9","review_type":"quick"}'
```

## 📊 Monitoring & Analytics

### **Health Monitoring**

```bash
# Service health checks
curl http://localhost:9200/health    # Inference service
curl http://localhost:9100/health    # Backend service
curl http://localhost:9000/          # Frontend service
```

### **Performance Metrics**

- **Response Times**: Track API latency and throughput
- **Model Accuracy**: Monitor review quality and user feedback
- **Resource Usage**: CPU, memory, and network utilization
- **Error Rates**: Track failures and timeout rates

### **Logging**

```bash
# Centralized logging
docker compose logs -f | tee contract-platform.log

# Structured logging format
{
  "timestamp": "2024-01-01T12:00:00Z",
  "service": "inference-service",
  "level": "INFO",
  "message": "Contract review completed",
  "model_id": "sftj-qbplmzw9",
  "response_time_ms": 1250
}
```

## 🔮 Roadmap & Future Enhancements

### **Planned Features**

- [ ] **Document Parsing**: PDF/Word upload and extraction
- [ ] **Batch Processing**: Bulk contract analysis
- [ ] **User Management**: Authentication and role-based access
- [ ] **Advanced Analytics**: Detailed reporting and insights
- [ ] **Model Comparison**: A/B testing different fine-tuned models
- [ ] **Export Features**: PDF reports and integration APIs

### **Performance Improvements**

- [ ] **Caching Layer**: Redis for frequently accessed data
- [ ] **Load Balancing**: Multiple inference service instances
- [ ] **Model Optimization**: Quantization and acceleration
- [ ] **Progressive Web App**: Offline capabilities

## 📝 Contributing

### **Development Setup**

```bash
# Fork repository and create feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
cd frontend && npm install
cd backend && npm install
cd inference-service && pip install -r requirements.txt

# Make changes and test
npm test                    # Frontend tests
python test_*.py           # Backend tests
docker compose up -d       # Integration testing
```

### **Code Standards**

- **Frontend**: ESLint + Prettier for React/JavaScript
- **Backend**: Black + Flake8 for Python
- **Documentation**: Update README for new features
- **Testing**: Add tests for new functionality

### **Pull Request Process**

1. Create detailed PR description
2. Include screenshots for UI changes
3. Ensure all tests pass
4. Update documentation as needed
5. Request review from maintainers

## 📞 Support & Resources

- **Documentation**: Component-specific READMEs in each directory
- **API Docs**: OpenAPI specs available at `/docs` endpoints
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

---

**🏆 Contract Review Platform v2.1.0**  
**⚡ Powered by Fireworks.ai & Llama 3.2**  
**🔧 Built with React, FastAPI, Docker**  
**📅 Last Updated: January 2025** 