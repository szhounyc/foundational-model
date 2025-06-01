FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories for contract review system
RUN mkdir -p uploads database data/pdfs data/processed models

# Expose port
EXPOSE 9100

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9100"] 