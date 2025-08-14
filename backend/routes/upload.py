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
    contents = await file.read()

    # Generate unique contract ID using hash of content
    contract_id = hashlib.md5(contents).hexdigest()
    output_path = f"backend/storage/extracted_texts/{contract_id}.txt"
    text_data = extract_text_from_pdf(contents)
    s3.put_object(
    Bucket=S3_BUCKET,
    Key=f"extracted_texts/{contract_id}.txt",
    Body=text_data.encode("utf-8")
)
    # Avoid reprocessing if already saved
  #  if not os.path.exists(output_path):
  #      text = extract_text_from_pdf(contents)
  #      os.makedirs("backend/storage/extracted_texts", exist_ok=True)
#        with open(output_path, "w") as f:
  #          f.write(text)

  #  return {"contract_id": contract_id, "message": "Text extracted and saved."}
