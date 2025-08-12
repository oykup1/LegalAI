import fitz  # PyMuPDF
from io import BytesIO

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text.strip()
