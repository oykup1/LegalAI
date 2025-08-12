import os
import json
import ollama
from fastapi import APIRouter, HTTPException
from backend.services.process_text import (
    split_into_chunks,
    generate_summary,
    build_and_save_faiss_index,
    load_faiss_index_and_chunks,
    search_faiss_index,
)

router = APIRouter()

RAW_TEXT_DIR = "backend/storage/extracted_texts"
PROCESSED_DIR = "backend/storage/processed_contracts"
os.makedirs(PROCESSED_DIR, exist_ok=True)

@router.post("/process/{contract_id}")
def process_contract_route(contract_id: str):
    try:
        return process_contract_logic(contract_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

def process_contract_logic(contract_id: str):
    # Load extracted text
    text_path = os.path.join(RAW_TEXT_DIR, f"{contract_id}.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"No extracted text found for contract ID: {contract_id}")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text
    chunks = split_into_chunks(text)

    # Save FAISS index + metadata
    build_and_save_faiss_index(chunks, contract_id)

    # Summarize each chunk
    processed_chunks = [{"chunk": chunk, "summary": generate_summary(chunk)} for chunk in chunks]

    # Save summaries
    output_path = os.path.join(PROCESSED_DIR, f"{contract_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, indent=2)

    return {"status": "success", "chunks_processed": len(processed_chunks)}


# -------------------- QUERY ENDPOINT ------------------------

@router.post("/query/{contract_id}")
def query_contract(contract_id: str, query: str):
    # Load FAISS index and chunks
    try:
        index, chunks = load_faiss_index_and_chunks(contract_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="FAISS index or metadata not found. Process the contract first.")

    # Search for relevant chunks
    relevant_chunks = search_faiss_index(query, index, chunks)
    combined_context = "\n".join(relevant_chunks)

    # Ask Ollama using retrieved context
    response = ollama.chat(
        model='llama3:8b',
        messages=[
            {'role': 'system', 'content': 'Answer based only on the following contract clauses:'},
            {'role': 'user', 'content': f"{combined_context}\n\nQuestion: {query}"}
        ]
    )

    return {"answer": response['message']['content']}
