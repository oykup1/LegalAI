# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Copy the app code
COPY . .

ENV OLLAMA_HOST=http://localhost:11434
# Expose FastAPI port
EXPOSE 8000

ENV S3_BUCKET_NAME=my-legal-ai-app

# Command to run the app
CMD ["uvicorn", "main.main:app", "--host", "0.0.0.0", "--port", "8000"]
