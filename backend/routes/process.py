import os
import json
from fastapi import APIRouter, HTTPException
from backend.ollama_client import ollama_client
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
    text_path = os.path.join(RAW_TEXT_DIR, f"{contract_id}.txt")
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"No extracted text found for contract ID: {contract_id}")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_into_chunks(text)
    build_and_save_faiss_index(chunks, contract_id)

    processed_chunks = [{"chunk": chunk, "summary": generate_summary(chunk)} for chunk in chunks]

    # Saving summaries locally is still your choice; can be skipped or changed to S3 if you want
    output_path = os.path.join(PROCESSED_DIR, f"{contract_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, indent=2)

    return {"status": "success", "chunks_processed": len(processed_chunks)}

@router.post("/query/{contract_id}")
def query_contract(contract_id: str, query: str):
    try:
        index, chunks = load_faiss_index_and_chunks(contract_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="FAISS index or metadata not found. Process the contract first.")

    relevant_chunks = search_faiss_index(query, index, chunks)
    combined_context = "\n".join(relevant_chunks)

    response = ollama_client.chat(
        model='llama3:8b',
        messages=[
            {'role': 'system', 'content': 'Answer based only on the following contract clauses:'},
            {'role': 'user', 'content': f"{combined_context}\n\nQuestion: {query}"}
        ]
    )

    return {"answer": response['message']['content']}
