import os
import re
import json
import numpy as np
import faiss
import tiktoken
import boto3
from sentence_transformers import SentenceTransformer
from backend.ollama_client import ollama_client

# AWS S3 setup - change bucket and region
S3_BUCKET_NAME = "my-legal-ai-app-bucket"
AWS_REGION = "us-east-1"
s3_client = boto3.client("s3", region_name=AWS_REGION)

# Directories still needed locally if you want caching/fallback (optional)
INDEX_DIR = "backend/storage/indexes"
META_DIR = "backend/storage/metadata"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

tokenizer = tiktoken.get_encoding('cl100k_base')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Your existing split_into_chunks and generate_summary unchanged
def split_into_chunks(text: str, max_tokens=512) -> list[str]:
    pattern = r"(?=\n?\d+\.\s[A-Z])"
    clauses = re.split(pattern, text)
    clauses = [c.strip() for c in clauses if c.strip()]
    if len(clauses) > 1:
        return clauses
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = tokenizer.encode(para)
        if current_tokens + len(para_tokens) <= max_tokens:
            current_chunk += "\n" + para
            current_tokens += len(para_tokens)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
            current_tokens = len(para_tokens)
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embedding(text: str) -> np.ndarray:
    return embedding_model.encode(text, convert_to_numpy=True).astype('float32')

def generate_summary(text: str) -> str:
    response = ollama_client.chat(
        model='llama3.2:1b',
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a legal contract analyzer. Given a contract clause or section, extract key structured data.\n\n'
                    'Respond ONLY with a single raw JSON object and NOTHING else — no explanations, no markdown, no ```json blocks.\n\n'
                    'Format:\n'
                    '{\n'
                    '  "clause_type": "...",\n'
                    '  "parties_involved": [...],\n'
                    '  "summary": "...",\n'
                    '  "biased_toward": "Client" | "Provider" | "Neutral",\n'
                    '  "risks": [...],\n'
                    '  "obligations": [...],\n'
                    '  "duration": "...",\n'
                    '  "is_termination_clause": true/false,\n'
                    '  "is_confidentiality_clause": true/false\n'
                    '}'
                )
            },
            {'role': 'user', 'content': text}
        ]
    )
    return response['message']['content']

# --- Modified to save FAISS index and chunks JSON TO S3 ---
def build_and_save_faiss_index(chunks: list[str], contract_id: str):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Serialize FAISS index to bytes
    index_bytes = faiss.serialize_index(index)
    # Upload FAISS index bytes to S3
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"indexes/{contract_id}.faiss", Body=index_bytes)

    # Save chunk metadata JSON to bytes
    chunks_json_bytes = json.dumps(chunks).encode("utf-8")
    s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"metadata/{contract_id}.json", Body=chunks_json_bytes)

# --- Modified to load FAISS index and chunk metadata FROM S3 ---
def load_faiss_index_and_chunks(contract_id: str):
    try:
        index_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"indexes/{contract_id}.faiss")
        index_bytes = index_obj['Body'].read()
        index = faiss.deserialize_index(index_bytes)

        meta_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"metadata/{contract_id}.json")
        chunks_json = meta_obj['Body'].read()
        chunks = json.loads(chunks_json.decode("utf-8"))

        return index, chunks
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"No FAISS index or metadata found in S3 for contract_id={contract_id}")

def search_faiss_index(query_text: str, index, chunks, top_k=3):
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]
