#!/bin/bash

# LLM Fine-tuning Platform Setup Script
echo "🚀 Setting up LLM Fine-tuning Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed (modern version)
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/pdfs
mkdir -p data/processed
mkdir -p models
mkdir -p model-trainer/checkpoints

# Set permissions
chmod -R 755 data/
chmod -R 755 models/
chmod -R 755 model-trainer/checkpoints/

# Check system architecture
ARCH=$(uname -m)
OS=$(uname -s)

echo "🖥️  Detected system: $OS $ARCH"

# Special handling for Apple Silicon
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon (M-series chip)"
    
    # Check if it's M4 Max specifically
    CPU_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    if [[ "$CPU_INFO" == *"M4"* ]]; then
        echo "⚡ Detected Apple M4 Max - GPU acceleration will be enabled"
    fi
    
    # Create Apple Silicon specific docker-compose override
    cat > docker-compose.override.yml << EOF
version: '3.8'
services:
  model-trainer:
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
    environment:
      - PYTORCH_ENABLE_MPS_FALLBACK=1
      - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  
  inference-service:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
    environment:
      - PYTORCH_ENABLE_MPS_FALLBACK=1
      - PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
EOF
    echo "✅ Created Apple Silicon optimizations"
fi

# Install frontend dependencies if Node.js is available
if command -v npm &> /dev/null; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✅ Frontend dependencies installed"
else
    echo "⚠️  Node.js not found. Frontend dependencies will be installed in Docker."
fi

# Create environment file
echo "⚙️  Creating environment configuration..."
cat > .env << EOF
# LLM Fine-tuning Platform Configuration

# Backend Configuration
DATABASE_URL=sqlite:///./app.db
PDF_PROCESSOR_URL=http://pdf-processor:8001
TRAINER_URL=http://model-trainer:8002
INFERENCE_URL=http://inference-service:8003

# Frontend Configuration
REACT_APP_BACKEND_URL=http://localhost:8000
REACT_APP_PDF_PROCESSOR_URL=http://localhost:8001
REACT_APP_TRAINER_URL=http://localhost:8002
REACT_APP_INFERENCE_URL=http://localhost:8003

# Ray Configuration
RAY_DISABLE_IMPORT_WARNING=1

# Hardware Detection
PYTORCH_ENABLE_MPS_FALLBACK=1
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
EOF

# Create sample data directory structure
echo "📄 Creating sample data structure..."
cat > data/README.md << EOF
# Data Directory

This directory contains:

- \`pdfs/\` - Uploaded PDF files
- \`processed/\` - Processed text data from PDFs

## Usage

1. Upload PDF files through the web interface
2. Process them to extract text chunks
3. Use processed data for model training
EOF

# Create models directory structure
cat > models/README.md << EOF
# Models Directory

This directory contains trained/fine-tuned models.

Each model is stored in its own subdirectory with:
- \`config.json\` - Model configuration
- \`pytorch_model.bin\` - Model weights
- \`tokenizer.json\` - Tokenizer configuration
- \`tokenizer_config.json\` - Tokenizer settings

## Usage

Trained models are automatically saved here after fine-tuning
and can be loaded for inference.
EOF

echo "🔧 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start the services: docker compose up -d"
echo "2. Access the web interface: http://localhost:3000"
echo "3. Check service health: http://localhost:8000/health"
echo ""
echo "Services will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- PDF Processor: http://localhost:8001"
echo "- Model Trainer: http://localhost:8002"
echo "- Inference Service: http://localhost:8003"
echo "- Ray Dashboard: http://localhost:8265"
echo ""
echo "For Apple M4 Max users: GPU acceleration is automatically configured!"
echo ""
echo "Happy fine-tuning! 🤖" 