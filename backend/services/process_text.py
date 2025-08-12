import os
import re
import json
import numpy as np
import faiss
import tiktoken
import ollama
from sentence_transformers import SentenceTransformer

# Directories for FAISS index and chunk metadata
INDEX_DIR = "backend/storage/indexes"
META_DIR = "backend/storage/metadata"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)


# Explicitly set the base URL so Docker can talk to the host
ollama_host = "http://host.docker.internal:11434"  # This special DNS points from Docker to your host machine
ollama_client = ollama.Client(host=ollama_host)

# Initialize tokenizer and embedding model
tokenizer = tiktoken.get_encoding('cl100k_base')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def split_into_chunks(text: str, max_tokens=512) -> list[str]:
    # Pattern: Numbered clauses like "1. Title", "2. Title", etc.
    pattern = r"(?=\n?\d+\.\s[A-Z])"
    clauses = re.split(pattern, text)
    clauses = [c.strip() for c in clauses if c.strip()]
    if len(clauses) > 1:
        return clauses  # Clause-based splitting worked

    # Fallback: Token-based chunking using sentence/paragraph boundaries
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
        model='llama3.2:1b',  # You mentioned llama3.2:1b — preserved here
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
            {
                'role': 'user',
                'content': text
            }
        ]
    )
    return response['message']['content']


def build_and_save_faiss_index(chunks: list[str], contract_id: str):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index to disk
    faiss.write_index(index, os.path.join(INDEX_DIR, f"{contract_id}.faiss"))

    # Save the chunk metadata
    with open(os.path.join(META_DIR, f"{contract_id}.json"), "w") as f:
        json.dump(chunks, f)


def load_faiss_index_and_chunks(contract_id: str):
    index_path = os.path.join(INDEX_DIR, f"{contract_id}.faiss")
    meta_path = os.path.join(META_DIR, f"{contract_id}.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing index or metadata for contract_id={contract_id}")

    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        chunks = json.load(f)

    return index, chunks


def search_faiss_index(query_text: str, index, chunks, top_k=3):
    query_embedding = embedding_model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]
