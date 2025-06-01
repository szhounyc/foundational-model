#!/bin/bash

# LLM Fine-tuning Platform Test Script
echo "🧪 Testing LLM Fine-tuning Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test service health
test_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Testing $service_name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    
    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL (HTTP $response)${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to test API endpoint with JSON response
test_api_endpoint() {
    local endpoint_name=$1
    local url=$2
    local method=${3:-GET}
    
    echo -n "Testing $endpoint_name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s "$url" 2>/dev/null)
    else
        response=$(curl -s -X "$method" "$url" 2>/dev/null)
    fi
    
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((TESTS_FAILED++))
    fi
}

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

echo ""
echo "🔍 Testing Service Health..."

# Test all services
test_service "Frontend" "http://localhost:3000"
test_service "Backend API" "http://localhost:8000/health"
test_service "PDF Processor" "http://localhost:8001/health"
test_service "Model Trainer" "http://localhost:8002/health"
test_service "Inference Service" "http://localhost:8003/health"

echo ""
echo "🔍 Testing API Endpoints..."

# Test backend endpoints
test_api_endpoint "Backend Root" "http://localhost:8000/"
test_api_endpoint "Backend PDFs List" "http://localhost:8000/pdfs"
test_api_endpoint "Backend Training Jobs" "http://localhost:8000/training/jobs"
test_api_endpoint "Backend Models List" "http://localhost:8000/models"

# Test PDF processor endpoints
test_api_endpoint "PDF Processor Root" "http://localhost:8001/"
test_api_endpoint "PDF Processor Files" "http://localhost:8001/files"

# Test model trainer endpoints
test_api_endpoint "Model Trainer Root" "http://localhost:8002/"
test_api_endpoint "Model Trainer Jobs" "http://localhost:8002/jobs"

# Test inference service endpoints
test_api_endpoint "Inference Service Root" "http://localhost:8003/"
test_api_endpoint "Inference Service Models" "http://localhost:8003/models"

echo ""
echo "🔍 Testing Docker Services..."

# Check if all containers are running
services=("frontend" "backend" "pdf-processor" "model-trainer" "inference-service" "redis")

for service in "${services[@]}"; do
    echo -n "Checking $service container... "
    
    if docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${GREEN}✅ RUNNING${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ NOT RUNNING${NC}"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "🔍 Testing File System..."

# Check if directories exist
directories=("data/pdfs" "data/processed" "models" "model-trainer/checkpoints")

for dir in "${directories[@]}"; do
    echo -n "Checking $dir directory... "
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅ EXISTS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ MISSING${NC}"
        ((TESTS_FAILED++))
    fi
done

echo ""
echo "🔍 Testing Hardware Detection..."

# Test hardware detection
echo -n "Testing hardware detection... "
ARCH=$(uname -m)
OS=$(uname -s)

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    CPU_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    if [[ "$CPU_INFO" == *"M4"* ]]; then
        echo -e "${GREEN}✅ Apple M4 Max detected${NC}"
    else
        echo -e "${YELLOW}⚠️  Apple Silicon detected (not M4 Max)${NC}"
    fi
    ((TESTS_PASSED++))
elif [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    else
        echo -e "${YELLOW}⚠️  Linux CPU detected${NC}"
    fi
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  Generic hardware detected${NC}"
    ((TESTS_PASSED++))
fi

echo ""
echo "🔍 Testing Environment Configuration..."

# Check if .env file exists
echo -n "Checking .env file... "
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ EXISTS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ MISSING${NC}"
    ((TESTS_FAILED++))
fi

# Check if docker-compose.override.yml exists for Apple Silicon
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo -n "Checking Apple Silicon optimizations... "
    if [ -f "docker-compose.override.yml" ]; then
        echo -e "${GREEN}✅ CONFIGURED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ MISSING${NC}"
        ((TESTS_FAILED++))
    fi
fi

echo ""
echo "📊 Test Results:"
echo "=================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed! The platform is ready to use.${NC}"
    echo ""
    echo "You can now:"
    echo "1. Access the web interface: http://localhost:3000"
    echo "2. Upload PDF files for processing"
    echo "3. Start training models"
    echo "4. Run inference on trained models"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed. Please check the logs and fix issues.${NC}"
    echo ""
    echo "Debugging tips:"
    echo "1. Check service logs: docker-compose logs [service-name]"
    echo "2. Restart services: docker-compose restart"
    echo "3. Rebuild containers: docker-compose up --build"
    exit 1
fi 