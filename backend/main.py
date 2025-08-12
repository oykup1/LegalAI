from fastapi import FastAPI
from routes import upload, process

app = FastAPI()

app.include_router(upload.router, prefix="/api")
app.include_router(process.router, prefix="/api")
