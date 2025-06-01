# Contract Review System - Docker Setup

This document provides instructions for running the complete Contract Review System using Docker containers.

## 🏗️ Architecture

The system consists of three main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │───▶│   Backend API   │───▶│ Inference Service│
│   (React App)   │    │   (FastAPI)     │    │   (ML Model)    │
│   Port: 9000    │    │   Port: 9100    │    │   Port: 9200    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **Frontend UI** (`frontend/`)
   - React application with Material-UI
   - Contract upload interface
   - Review results dashboard
   - Review history management

2. **Backend API** (`backend/`)
   - FastAPI server
   - PDF processing and text extraction
   - Database management (SQLite)
   - API endpoints for contract operations

3. **Inference Service** (`inference-service/`)
   - Fine-tuned model server
   - Contract review AI processing
   - Model loading and inference

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available
- Your fine-tuned model in `models/sftj-s1xkr35z/` directory

### Start the System

```bash
# Make scripts executable (first time only)
chmod +x start-contract-stack.sh stop-contract-stack.sh

# Start the complete stack
./start-contract-stack.sh
```

The script will:
1. Check Docker availability
2. Verify required directories and files
3. Build and start all services
4. Wait for services to be healthy
5. Display access URLs

### Stop the System

```bash
# Stop all services
./stop-contract-stack.sh

# Stop and cleanup resources
./stop-contract-stack.sh --cleanup
```

## 📋 Service Details

### Service URLs

- **Frontend**: http://localhost:9000
- **Backend API**: http://localhost:9100
- **Inference Service**: http://localhost:9200

### Health Check Endpoints

- **Backend**: http://localhost:9100/health
- **Inference**: http://localhost:9200/health

### API Documentation

- **Backend Swagger UI**: http://localhost:9100/docs
- **Backend ReDoc**: http://localhost:9100/redoc

## 🔧 Configuration

### Environment Variables

The system uses the following environment variables:

#### Frontend
- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:9100)

#### Backend
- `INFERENCE_SERVICE_URL`: Inference service URL (default: http://inference-service:8003)
- `DATABASE_PATH`: SQLite database path (default: /app/database/contracts.db)
- `UPLOAD_DIR`: File upload directory (default: /app/uploads)

#### Inference Service
- `PYTHONPATH`: Python path (default: /app)
- `TOKENIZERS_PARALLELISM`: Tokenizer parallelism (default: false)
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: 0)

### Volume Mounts

- `./models:/app/models:ro` - Model files (read-only)
- `./dataset:/app/dataset:ro` - Dataset files (read-only)
- `./backend/uploads:/app/uploads` - Uploaded contract files
- `./backend/database:/app/database` - SQLite database

## 🛠️ Manual Docker Commands

If you prefer to run Docker commands manually:

### Build Images

```bash
# Build all images
docker compose -f docker-compose.full-stack.yml build

# Build specific service
docker compose -f docker-compose.full-stack.yml build frontend
docker compose -f docker-compose.full-stack.yml build backend
docker compose -f docker-compose.full-stack.yml build inference-service
```

### Start Services

```bash
# Start all services
docker compose -f docker-compose.full-stack.yml up -d

# Start specific service
docker compose -f docker-compose.full-stack.yml up -d frontend
```

### View Logs

```bash
# View all logs
docker compose -f docker-compose.full-stack.yml logs -f

# View specific service logs
docker compose -f docker-compose.full-stack.yml logs -f backend
docker compose -f docker-compose.full-stack.yml logs -f inference-service
```

### Stop Services

```bash
# Stop all services
docker compose -f docker-compose.full-stack.yml down

# Stop and remove volumes
docker compose -f docker-compose.full-stack.yml down -v
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :9000  # or :9100, :9200
   
   # Kill the process
   kill -9 <PID>
   ```

2. **Model Loading Issues**
   - Ensure your model is in `models/sftj-s1xkr35z/552ca5/zlg-re-fm-sl-mntn/checkpoint/`
   - Check inference service logs: `docker compose -f docker-compose.full-stack.yml logs inference-service`

3. **Memory Issues**
   - Ensure you have at least 8GB RAM available
   - Monitor Docker resource usage: `docker stats`

4. **Network Issues**
   - Check if containers can communicate: `docker network ls`
   - Restart Docker if needed

### Debug Commands

```bash
# Check container status
docker compose -f docker-compose.full-stack.yml ps

# Execute commands in containers
docker exec -it contract-backend bash
docker exec -it contract-inference-service bash
docker exec -it contract-frontend sh

# Check container logs
docker logs contract-backend
docker logs contract-inference-service
docker logs contract-frontend

# Check network connectivity
docker exec contract-backend curl http://inference-service:8003/health
```

## 📊 Monitoring

### Container Health

All services include health checks that monitor:
- Service availability
- API responsiveness
- Model loading status

### Resource Usage

Monitor resource usage with:
```bash
# Real-time stats
docker stats

# Container resource limits
docker compose -f docker-compose.full-stack.yml config
```

## 🔄 Development Workflow

### Making Changes

1. **Frontend Changes**
   ```bash
   # Rebuild and restart frontend
   docker compose -f docker-compose.full-stack.yml build frontend
   docker compose -f docker-compose.full-stack.yml up -d frontend
   ```

2. **Backend Changes**
   ```bash
   # Rebuild and restart backend
   docker compose -f docker-compose.full-stack.yml build backend
   docker compose -f docker-compose.full-stack.yml up -d backend
   ```

3. **Inference Service Changes**
   ```bash
   # Rebuild and restart inference service
   docker compose -f docker-compose.full-stack.yml build inference-service
   docker compose -f docker-compose.full-stack.yml up -d inference-service
   ```

### Database Management

```bash
# Access SQLite database
docker exec -it contract-backend sqlite3 /app/database/contracts.db

# Backup database
docker cp contract-backend:/app/database/contracts.db ./backup-$(date +%Y%m%d).db

# View uploaded files
docker exec -it contract-backend ls -la /app/uploads/
```

## 🚨 Production Considerations

For production deployment, consider:

1. **Security**
   - Use environment files for sensitive data
   - Implement proper authentication
   - Use HTTPS with SSL certificates
   - Restrict network access

2. **Scalability**
   - Use external databases (PostgreSQL)
   - Implement load balancing
   - Use container orchestration (Kubernetes)
   - Add monitoring and logging

3. **Data Persistence**
   - Use external volumes for data
   - Implement backup strategies
   - Use cloud storage for files

4. **Performance**
   - Optimize model loading
   - Implement caching
   - Use CDN for frontend assets
   - Monitor resource usage

## 📝 Support

If you encounter issues:

1. Check the logs: `docker compose -f docker-compose.full-stack.yml logs`
2. Verify all required files are present
3. Ensure Docker has sufficient resources
4. Check network connectivity between containers

For additional help, refer to the individual service documentation in their respective directories. 