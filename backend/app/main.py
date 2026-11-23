import os
import shutil
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.database import init_db, SessionLocal
from app.rag_service import ingest_pdf_to_db
from pydantic import BaseModel
from app.rag_service import ingest_pdf_to_db, generate_healthcare_content

app = FastAPI(title="Healthcare Content & Clinic Analytics AI API")

# Izinkan koneksi dari frontend Next.js nantinya
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def health_check():
    return {"status": "online", "service": "Clinic AI Assistant API"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file format PDF yang didukung.")

    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = os.path.join("temp_uploads", file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_pdf_to_db(temp_path, file.filename, db)
        return {"message": "Dokumen berhasil diproses ke vector database", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

class ContentRequest(BaseModel):
    topic: str
    format_type: str = "Instagram Carousel"  # Contoh: "Instagram Carousel", "Reels Script", "Caption Edukasi"

@app.post("/generate-content")
def create_content(request: ContentRequest, db: Session = Depends(get_db)):
    try:
        result = generate_healthcare_content(
            topic=request.topic,
            format_type=request.format_type,
            db=db
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))