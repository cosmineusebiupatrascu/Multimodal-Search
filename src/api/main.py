import io
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.search import SearchEngine
from src.vector_db import QdrantDB

search_engine = None
database = None


async def lifespan(app: FastAPI):
    global search_engine, database

    search_engine = SearchEngine()
    database = QdrantDB()

    yield

    search_engine = None
    database = None

app = FastAPI(
    title="Multimodal Search API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

IMAGE_DIR = Path("data/raw_images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(IMAGE_DIR)), name="static")


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "multimodal-search-api"
    }


@app.get("/stats", tags=["Monitoring"])
def get_stats():
    try:
        return database.get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/text", tags=["Search"])
def search_by_text(
        query: str = Form(...),
        limit: int = Form(5)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")

    try:
        return search_engine.search_by_text(query_text=query, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search/image", tags=["Search"])
async def search_by_image(
        file: UploadFile = File(...),
        limit: int = Form(5)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return search_engine.search_by_image(query_image=image, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")


@app.post("/ingest/image", tags=["Insert"])
async def ingest_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid format. Only images accepted.")

    file_path = IMAGE_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"/static/{file.filename}"

    try:
        image = Image.open(file_path).convert("RGB")
        vector = search_engine.encoder.encode_image(image)

        payload = {
            "image_url": image_url,
            "filename": file.filename
        }

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file.filename))
        database.upsert_vector(point_id=point_id, vector=vector, payload=payload)

        return {
            "status": "success",
            "image_url": image_url
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete/points", tags=["Management"])
def delete_points(payload: list[str] = Body(...)):
    if not payload:
        raise HTTPException(status_code=400, detail="IDs list cannot be empty")

    try:
        database.delete_points(point_ids=payload)
        return {
            "status": "success",
            "deleted_count": len(payload)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete points: {str(e)}")



