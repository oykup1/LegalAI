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
                    """
You are a highly precise legal contract analyst. You will be given a single clause or section of a contract. 
Your task is to extract key structured information in JSON format only — NO explanations, no markdown, no code blocks. 
Always return a single valid JSON object. Fill every field; if information is missing, use null for strings, empty lists for lists, and false for booleans.

The JSON format must be exactly:

{
  "clause_type": "<type of clause, e.g., 'Payment Terms'>",
  "parties_involved": ["<party1>", "<party2>", ...],
  "summary": "<plain English summary of clause>",
  "biased_toward": "Client" | "Provider" | "Neutral",
  "risks": ["<risk description>", ...],
  "obligations": ["<obligation description>", ...],
  "duration": "<duration or null>",
  "is_termination_clause": true | false,
  "is_confidentiality_clause": true | false
}

Rules and guidance:

1. Focus strictly on the clause you are given. Do NOT include information from other clauses or contracts.
2. Summarize obligations in clear, concise plain English, one obligation per list item.
3. If the clause specifies deadlines, durations, or effective periods, include in 'duration'.
4. Determine bias based on which party benefits or bears risk; use 'Neutral' if neither clearly benefits.
5. Risks should describe anything that could negatively impact a party.
6. For any field that cannot be inferred, use null, empty list, or false as appropriate.
7. Use consistent naming of parties and obligations so that this can later be used to compare clauses across contracts.
8. Always return **valid JSON** with all fields present.
9. Treat each clause independently but maintain consistency in party names across clauses if they are mentioned repeatedly.
"""
                )
            },
            {'role': 'user', 'content': text}
        ]
    )
    return json.loads(response['message']['content'])


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
