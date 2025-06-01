#!/bin/bash

# Contract Review Stack Startup Script
# This script starts the full contract review system with Frontend, Backend, and Inference Service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "🚀 Contract Review Stack Startup"
    echo "=================================================="
    echo -e "${NC}"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! command -v docker-compose > /dev/null 2>&1 && ! docker compose version > /dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Check required directories and files
check_requirements() {
    print_status "Checking requirements..."
    
    # Check for required directories
    required_dirs=("frontend" "backend" "inference-service" "models")
    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            print_error "Required directory '$dir' not found"
            exit 1
        fi
    done
    
    # Check for Docker Compose file
    if [ ! -f "docker-compose.full-stack.yml" ]; then
        print_error "Docker Compose file 'docker-compose.full-stack.yml' not found"
        exit 1
    fi
    
    # Check for models
    if [ ! -d "models/sftj-s1xkr35z" ]; then
        print_warning "Model directory 'models/sftj-s1xkr35z' not found. Make sure your model is available."
    fi
    
    print_success "All requirements satisfied"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    # Create upload and database directories
    mkdir -p backend/uploads
    mkdir -p backend/database
    
    print_success "Directories created"
}

# Stop any existing containers
stop_existing() {
    print_status "Stopping any existing containers..."
    
    # Use docker-compose if available, otherwise use docker compose
    if command -v docker-compose > /dev/null 2>&1; then
        docker-compose -f docker-compose.full-stack.yml down --remove-orphans 2>/dev/null || true
    else
        docker compose -f docker-compose.full-stack.yml down --remove-orphans 2>/dev/null || true
    fi
    
    print_success "Existing containers stopped"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Use docker-compose if available, otherwise use docker compose
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Build and start services
    $COMPOSE_CMD -f docker-compose.full-stack.yml up --build -d
    
    print_success "Services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for inference service
    print_status "Waiting for inference service (this may take a few minutes for model loading)..."
    timeout=300  # 5 minutes
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -s http://localhost:9200/health > /dev/null 2>&1; then
            print_success "Inference service is ready"
            break
        fi
        sleep 5
        counter=$((counter + 5))
        if [ $((counter % 30)) -eq 0 ]; then
            print_status "Still waiting for inference service... ($counter/${timeout}s)"
        fi
    done
    
    if [ $counter -ge $timeout ]; then
        print_error "Inference service failed to start within $timeout seconds"
        return 1
    fi
    
    # Wait for backend
    print_status "Waiting for backend service..."
    timeout=60
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -s http://localhost:9100/health > /dev/null 2>&1; then
            print_success "Backend service is ready"
            break
        fi
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ $counter -ge $timeout ]; then
        print_error "Backend service failed to start within $timeout seconds"
        return 1
    fi
    
    # Wait for frontend
    print_status "Waiting for frontend service..."
    timeout=60
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -s http://localhost:9000 > /dev/null 2>&1; then
            print_success "Frontend service is ready"
            break
        fi
        sleep 2
        counter=$((counter + 2))
    done
    
    if [ $counter -ge $timeout ]; then
        print_error "Frontend service failed to start within $timeout seconds"
        return 1
    fi
}

# Show service status
show_status() {
    print_status "Service Status:"
    echo ""
    
    # Use docker-compose if available, otherwise use docker compose
    if command -v docker-compose > /dev/null 2>&1; then
        docker-compose -f docker-compose.full-stack.yml ps
    else
        docker compose -f docker-compose.full-stack.yml ps
    fi
    
    echo ""
    print_success "🌐 Frontend:        http://localhost:9000"
    print_success "🔧 Backend API:     http://localhost:9100"
    print_success "🤖 Inference API:   http://localhost:9200"
    echo ""
    print_success "📊 Health Checks:"
    print_success "   Frontend:        http://localhost:9000"
    print_success "   Backend:         http://localhost:9100/health"
    print_success "   Inference:       http://localhost:9200/health"
}

# Main execution
main() {
    print_header
    
    check_docker
    check_docker_compose
    check_requirements
    create_directories
    stop_existing
    start_services
    
    if wait_for_services; then
        echo ""
        print_success "🎉 Contract Review Stack is ready!"
        show_status
        echo ""
        print_status "To stop the stack, run: ./stop-contract-stack.sh"
        print_status "To view logs, run: docker-compose -f docker-compose.full-stack.yml logs -f"
    else
        print_error "Failed to start all services. Check the logs for details:"
        print_error "docker-compose -f docker-compose.full-stack.yml logs"
        exit 1
    fi
}

# Run main function
main "$@" 