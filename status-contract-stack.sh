#!/bin/bash

# Contract Review Stack Status Script
# This script checks the status of all services in the contract review system

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
    echo "📊 Contract Review Stack Status"
    echo "=================================================="
    echo -e "${NC}"
}

# Check service health
check_service_health() {
    local service_name="$1"
    local url="$2"
    local timeout=5
    
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        print_success "$service_name is healthy"
        return 0
    else
        print_error "$service_name is not responding"
        return 1
    fi
}

# Check container status
check_containers() {
    print_status "Checking container status..."
    echo ""
    
    containers=("contract-frontend" "contract-backend" "contract-inference-service")
    running_count=0
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "$container"; then
            status=$(docker ps --format "{{.Status}}" -f name="$container")
            ports=$(docker ps --format "{{.Ports}}" -f name="$container")
            print_success "$container: $status"
            echo "   Ports: $ports"
            running_count=$((running_count + 1))
        else
            print_error "$container: Not running"
        fi
        echo ""
    done
    
    echo "Running containers: $running_count/3"
    return $running_count
}

# Check service health endpoints
check_health_endpoints() {
    print_status "Checking service health endpoints..."
    echo ""
    
    healthy_count=0
    
    # Check Frontend
    if check_service_health "Frontend (http://localhost:9000)" "http://localhost:9000"; then
        healthy_count=$((healthy_count + 1))
    fi
    
    # Check Backend
    if check_service_health "Backend API (http://localhost:9100/health)" "http://localhost:9100/health"; then
        healthy_count=$((healthy_count + 1))
    fi
    
    # Check Inference Service
    if check_service_health "Inference Service (http://localhost:9200/health)" "http://localhost:9200/health"; then
        healthy_count=$((healthy_count + 1))
    fi
    
    echo ""
    echo "Healthy services: $healthy_count/3"
    return $healthy_count
}

# Check Docker Compose status
check_compose_status() {
    print_status "Docker Compose status..."
    echo ""
    
    if [ -f "docker-compose.full-stack.yml" ]; then
        # Use docker-compose if available, otherwise use docker compose
        if command -v docker-compose > /dev/null 2>&1; then
            docker-compose -f docker-compose.full-stack.yml ps
        else
            docker compose -f docker-compose.full-stack.yml ps
        fi
    else
        print_warning "Docker Compose file not found"
    fi
    echo ""
}

# Show resource usage
show_resource_usage() {
    print_status "Resource usage..."
    echo ""
    
    # Show Docker stats for contract services
    if docker ps -q -f name=contract- | grep -q .; then
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" $(docker ps -q -f name=contract-)
    else
        print_warning "No contract services running"
    fi
    echo ""
}

# Show service URLs
show_service_urls() {
    print_status "Service URLs:"
    echo ""
    print_success "🌐 Frontend:        http://localhost:9000"
    print_success "🔧 Backend API:     http://localhost:9100"
    print_success "🤖 Inference API:   http://localhost:9200"
    echo ""
    print_status "📚 API Documentation:"
    print_success "   Swagger UI:      http://localhost:9100/docs"
    print_success "   ReDoc:           http://localhost:9100/redoc"
    echo ""
    print_status "🏥 Health Checks:"
    print_success "   Backend:         http://localhost:9100/health"
    print_success "   Inference:       http://localhost:9200/health"
    echo ""
}

# Show logs option
show_logs_info() {
    print_status "To view logs:"
    echo ""
    echo "  All services:      docker-compose -f docker-compose.full-stack.yml logs -f"
    echo "  Frontend only:     docker-compose -f docker-compose.full-stack.yml logs -f frontend"
    echo "  Backend only:      docker-compose -f docker-compose.full-stack.yml logs -f backend"
    echo "  Inference only:    docker-compose -f docker-compose.full-stack.yml logs -f inference-service"
    echo ""
}

# Main execution
main() {
    print_header
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi
    
    check_containers
    container_count=$?
    
    check_health_endpoints
    health_count=$?
    
    check_compose_status
    show_resource_usage
    show_service_urls
    show_logs_info
    
    # Overall status
    if [ $container_count -eq 3 ] && [ $health_count -eq 3 ]; then
        print_success "🎉 All services are running and healthy!"
    elif [ $container_count -eq 3 ]; then
        print_warning "⚠️  All containers are running but some services are not healthy"
    elif [ $container_count -gt 0 ]; then
        print_warning "⚠️  Some services are running but not all"
        print_status "To start all services: ./start-contract-stack.sh"
    else
        print_error "❌ No contract review services are running"
        print_status "To start the stack: ./start-contract-stack.sh"
    fi
}

# Show help
show_help() {
    echo "Contract Review Stack Status Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "This script checks the status of all contract review services:"
    echo "  - Container status"
    echo "  - Health endpoints"
    echo "  - Resource usage"
    echo "  - Service URLs"
}

# Handle command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac 