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


# Install Python dependencies in a single layer + clean cache
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the app code
COPY . .

# Point FastAPI to the running Ollama server on the host
ENV OLLAMA_HOST=http://host.docker.internal:11434
# Expose FastAPI port
EXPOSE 8000

ENV S3_BUCKET_NAME=my-legal-ai-app-bucket

# Command to run the app
CMD ["uvicorn", "main.main:app", "--host", "0.0.0.0", "--port", "8000"]
