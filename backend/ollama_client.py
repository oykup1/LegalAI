import os
import ollama

# Read from environment, default to localhost for dev
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

ollama_client = ollama.Client(host=ollama_host)
