from fastapi import APIRouter, UploadFile, File
from backend.services.extract_text_from_pdf import extract_text_from_pdf
import hashlib
import os
import boto3
from io import BytesIO


router = APIRouter()
s3 = boto3.client("s3")
S3_BUCKET = os.environ["S3_BUCKET_NAME"]

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            return {"error": "Empty file"}, 400

        contract_id = hashlib.md5(contents).hexdigest()
        text_data = extract_text_from_pdf(contents)

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"extracted_texts/{contract_id}.txt",
            Body=text_data.encode("utf-8")
        )

        return {"contract_id": contract_id, "message": "Text extracted and saved."}

    except Exception as e:
        # Log e here if you have logging
        return {"error": str(e)}, 500
