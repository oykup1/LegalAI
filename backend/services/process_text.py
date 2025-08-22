import json
import tempfile
import boto3
import numpy as np
import faiss
import re
import os
import tiktoken
from sentence_transformers import SentenceTransformer
from backend.ollama_client import ollama_client

# AWS S3 client
s3 = boto3.client('s3')
S3_BUCKET = os.environ["S3_BUCKET_NAME"]

tokenizer = tiktoken.get_encoding('cl100k_base')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


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


def generate_summary(text: str) -> str:
    response = ollama_client.chat(
        model='llama3.2:3b-instruct-q4_0',
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


def build_and_save_faiss_index(chunks: list[str], contract_id: str):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save to temporary files and immediately upload to S3
    with tempfile.TemporaryDirectory() as tmpdir:
        local_index_path = f"{tmpdir}/{contract_id}.faiss"
        local_meta_path = f"{tmpdir}/{contract_id}.json"

        faiss.write_index(index, local_index_path)
        with open(local_meta_path, "w") as f:
            json.dump(chunks, f)

        s3.upload_file(local_index_path, S3_BUCKET, f"indexes/{contract_id}.faiss")
        s3.upload_file(local_meta_path, S3_BUCKET, f"metadata/{contract_id}.json")


def load_faiss_index_and_chunks(contract_id: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_index_path = f"{tmpdir}/{contract_id}.faiss"
        local_meta_path = f"{tmpdir}/{contract_id}.json"

        try:
            s3.download_file(S3_BUCKET, f"indexes/{contract_id}.faiss", local_index_path)
            s3.download_file(S3_BUCKET, f"metadata/{contract_id}.json", local_meta_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not find FAISS index or metadata in S3 for contract_id={contract_id}: {e}")

        index = faiss.read_index(local_index_path)
        with open(local_meta_path) as f:
            chunks = json.load(f)

    return index, chunks


def search_faiss_index(query_text: str, index, chunks, top_k=3):
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]
