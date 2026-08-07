import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
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




