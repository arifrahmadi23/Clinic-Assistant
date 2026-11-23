import os
from pypdf import PdfReader
from google import genai
from sqlalchemy.orm import Session
from app.database import DocumentChunk
from dotenv import load_dotenv
from google.genai import types
import json
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Memecah teks panjang menjadi beberapa chunk dengan overlap konteks."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # Abaikan potongan terlalu pendek

def get_embedding(text: str) -> list[float]:
    """Mengubah teks menjadi vektor 768 dimensi menggunakan Gemini."""
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768)
    )
    return response.embeddings[0].values

def ingest_pdf_to_db(file_path: str, filename: str, db: Session) -> dict:
    """Membaca PDF per halaman, melakukan chunking, embedding, dan simpan ke PostgreSQL."""
    reader = PdfReader(file_path)
    total_chunks = 0

    # Hapus data lama jika file dengan nama sama di-upload ulang (idempotent)
    db.query(DocumentChunk).filter(DocumentChunk.filename == filename).delete()

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        if not raw_text:
            continue

        chunks = chunk_text(raw_text)
        for chunk in chunks:
            vector = get_embedding(chunk)
            db_chunk = DocumentChunk(
                filename=filename,
                page_number=page_idx,
                content=chunk,
                embedding=vector
            )
            db.add(db_chunk)
            total_chunks += 1

    db.commit()
    return {
        "filename": filename,
        "total_pages": len(reader.pages),
        "total_chunks_saved": total_chunks
    }

def generate_healthcare_content(topic: str, format_type: str, db: Session, top_k: int = 3) -> dict:
    """Mencari konteks relevan dari pgvector dan menghasilkan konten medis bersitasi."""
    # 1. Ubah topik pertanyaan menjadi vektor
    query_vector = get_embedding(topic)

    # 2. Cari top_k chunk paling mirip menggunakan cosine_distance
    results = (
        db.query(DocumentChunk)
        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
        .all()
    )

    if not results:
        return {
            "error": "Belum ada dokumen referensi di database. Silakan unggah PDF terlebih dahulu."
        }

    # 3. Susun konteks beserta tag sumber untuk sitasi
    context_blocks = []
    sources = []
    for idx, chunk in enumerate(results, start=1):
        source_label = f"{chunk.filename} (Halaman {chunk.page_number})"
        context_blocks.append(f"[Sumber {idx}: {source_label}]\n{chunk.content}")
        if source_label not in sources:
            sources.append(source_label)

    combined_context = "\n\n---\n\n".join(context_blocks)

    # 4. System & User Prompt dengan aturan anti-halusinasi dan output JSON
    prompt = f"""
Anda adalah asisten Content Planner & Medical Copywriter profesional untuk klinik kesehatan.
Tugas Anda adalah membuat draf konten media sosial bertipe "{format_type}" dengan topik: "{topic}".

ATURAN KETAT:
1. Gunakan HANYA informasi medis yang tersedia pada KONTEKS DOKUMEN di bawah ini. Jangan menambahkan klaim medis di luar konteks.
2. Gunakan bahasa Indonesia yang empatik, edukatif, mudah dipahami pasien, dan profesional.
3. Kembalikan respons HANYA dalam format JSON valid dengan struktur persis berikut:
{{
  "judul_hook": "Kalimat pembuka yang menarik perhatian audiens",
  "isi_konten": "Isi utama caption atau naskah konten (gunakan poin-poin jika perlu)",
  "call_to_action": "Ajakan interaksi atau konsultasi ke klinik",
  "hashtags": ["#tag1", "#tag2", "#tag3"],
  "catatan_akurasi": "Penjelasan singkat bagaimana konten ini merujuk pada dokumen"
}}

KONTEKS DOKUMEN:
{combined_context}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    generated_json = json.loads(response.text)
    generated_json["sumber_referensi"] = sources
    return generated_json