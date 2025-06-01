#!/bin/bash

echo "🚀 Starting Enhanced Inference Service with Docker"
echo "=================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if models directory exists
if [ ! -d "./models" ]; then
    echo "❌ Models directory not found. Please ensure Fireworks models are in ./models/"
    exit 1
fi

# Check for Fireworks models
if [ ! -f "./models/*/checkpoint/adapter_config.json" ]; then
    echo "⚠️  Warning: No Fireworks models found in ./models/"
    echo "   Expected structure: ./models/*/checkpoint/adapter_config.json"
fi

echo "🔍 Building and starting enhanced inference service..."

# Build and start the service
docker-compose -f docker-compose.inference.yml up --build

echo "✅ Enhanced inference service started!"
echo "📡 Service URL: http://localhost:8003"
echo "📚 API Docs: http://localhost:8003/docs"
echo "🧪 Health Check: curl http://localhost:8003/health" 