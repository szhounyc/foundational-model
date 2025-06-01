#!/bin/bash

# Contract Review Stack Stop Script
# This script stops the full contract review system

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
    echo "🛑 Contract Review Stack Shutdown"
    echo "=================================================="
    echo -e "${NC}"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running."
        exit 1
    fi
    print_success "Docker is running"
}

# Stop services
stop_services() {
    print_status "Stopping contract review services..."
    
    # Use docker-compose if available, otherwise use docker compose
    if command -v docker-compose > /dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        COMPOSE_CMD="docker compose"
    fi
    
    # Stop and remove containers
    if [ -f "docker-compose.full-stack.yml" ]; then
        $COMPOSE_CMD -f docker-compose.full-stack.yml down --remove-orphans
        print_success "Services stopped"
    else
        print_warning "Docker Compose file not found, attempting to stop individual containers..."
        
        # Stop individual containers if they exist
        containers=("contract-frontend" "contract-backend" "contract-inference-service")
        for container in "${containers[@]}"; do
            if docker ps -q -f name="$container" | grep -q .; then
                print_status "Stopping $container..."
                docker stop "$container" > /dev/null 2>&1 || true
                docker rm "$container" > /dev/null 2>&1 || true
                print_success "$container stopped"
            fi
        done
    fi
}

# Clean up resources (optional)
cleanup_resources() {
    if [ "$1" = "--cleanup" ] || [ "$1" = "-c" ]; then
        print_status "Cleaning up Docker resources..."
        
        # Remove unused networks
        if docker network ls | grep -q "contract-review-network"; then
            print_status "Removing contract review network..."
            docker network rm contract-review-network > /dev/null 2>&1 || true
        fi
        
        # Remove unused images (optional - commented out by default)
        # print_status "Removing unused Docker images..."
        # docker image prune -f > /dev/null 2>&1 || true
        
        # Remove unused volumes (optional - commented out by default)
        # print_warning "Removing unused volumes (this will delete data)..."
        # docker volume prune -f > /dev/null 2>&1 || true
        
        print_success "Cleanup completed"
    fi
}

# Show remaining containers
show_status() {
    print_status "Checking for remaining contract review containers..."
    
    containers=("contract-frontend" "contract-backend" "contract-inference-service")
    running_containers=0
    
    for container in "${containers[@]}"; do
        if docker ps -q -f name="$container" | grep -q .; then
            print_warning "$container is still running"
            running_containers=$((running_containers + 1))
        fi
    done
    
    if [ $running_containers -eq 0 ]; then
        print_success "All contract review containers have been stopped"
    else
        print_warning "$running_containers container(s) are still running"
        print_status "You can force stop them with: docker stop \$(docker ps -q -f name=contract-)"
    fi
}

# Main execution
main() {
    print_header
    
    check_docker
    stop_services
    cleanup_resources "$1"
    show_status
    
    echo ""
    print_success "🎉 Contract Review Stack has been stopped!"
    echo ""
    print_status "To start the stack again, run: ./start-contract-stack.sh"
    
    if [ "$1" != "--cleanup" ] && [ "$1" != "-c" ]; then
        print_status "To stop and cleanup all resources, run: ./stop-contract-stack.sh --cleanup"
    fi
}

# Show help
show_help() {
    echo "Contract Review Stack Stop Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cleanup, -c    Also cleanup Docker networks and unused resources"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               Stop the contract review stack"
    echo "  $0 --cleanup     Stop the stack and cleanup resources"
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