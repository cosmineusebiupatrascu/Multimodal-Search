from pathlib import Path
import torch
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw_images"
QDRANT_DB_DIR = BASE_DIR / "qdrant_db"

MODEL_NAME = "openai/clip-vit-base-patch32"
VECTOR_SIZE = 512

COLLECTION_NAME = "multimodal_catalog"

QDRANT_HOST = os.getenv("QDRANT_HOST", None)
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
