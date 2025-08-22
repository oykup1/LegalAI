import os
import json
import boto3
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

RAW_TEXT_DIR = "backend/storage/extracted_texts"  # Keep locally for now
PROCESSED_DIR = "backend/storage/processed_contracts"
os.makedirs(PROCESSED_DIR, exist_ok=True)
s3 = boto3.client('s3')
S3_BUCKET = os.environ["S3_BUCKET_NAME"]

@router.post("/process/{contract_id}")
def process_contract_route(contract_id: str):
    # Download extracted text from S3
    try:
        # Download extracted text from S3
        s3_object = s3.get_object(
            Bucket=S3_BUCKET,
            Key=f"extracted_texts/{contract_id}.txt"
        )
        text = s3_object["Body"].read().decode("utf-8")

        # Process the contract
        return process_contract_logic(contract_id, text)

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Logs full stack trace in terminal
        raise HTTPException(status_code=500, detail=str(e))


def validate_clause_json(clause_json: dict) -> dict:
    defaults = {
        "clause_type": "",
        "parties_involved": [],
        "summary": "",
        "biased_toward": "Neutral",
        "risks": [],
        "obligations": [],
        "duration": None,
        "is_termination_clause": False,
        "is_confidentiality_clause": False
    }
    for key, val in defaults.items():
        if key not in clause_json or clause_json[key] is None:
            clause_json[key] = val
    return clause_json


def process_contract_logic(contract_id: str, text: str):
    chunks = split_into_chunks(text)
    build_and_save_faiss_index(chunks, contract_id)

    processed_chunks = []
    for chunk in chunks:
        clause_json = generate_summary(chunk)
        clause_json = validate_clause_json(clause_json)  # ensure all fields exist
        processed_chunks.append({
            "chunk": chunk,
            "summary": clause_json
        })

    # Save processed chunks to S3
    processed_key = f"processed_contracts/{contract_id}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=processed_key,
        Body=json.dumps(processed_chunks, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

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
        model='llama3.2:3b-instruct-q4_0',
        messages=[
            {'role': 'system', 'content': 'Answer based only on the following contract clauses:'},
            {'role': 'user', 'content': f"{combined_context}\n\nQuestion: {query}"}
        ]
    )

    return {"answer": response['message']['content']}
