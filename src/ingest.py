import os
from pathlib import Path
from tqdm import tqdm
import uuid

from src.encoder import CLIPEncoder
from src.vector_db import QdrantDB

NAMESPACE_MULTIMODAL = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def generate_uuid(path: Path):
    return str(uuid.uuid5(NAMESPACE_MULTIMODAL, str(path.resolve())))


def ingest_data(photos_dir: str = "data/raw_images"):
    path = Path(photos_dir)

    if not path.exists():
        print(f"No directory found: {path}")
        return

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")

    image_files = []
    for ext in extensions:
        image_files.extend(list(path.glob(ext)))

    if not image_files:
        print(f"No images found in directory: {path}")
        return

    print(f"{len(image_files)} found")

    encoder = CLIPEncoder()
    database = QdrantDB()

    for img_path in tqdm(image_files, desc="Processing images"):
        point_id = generate_uuid(img_path)
        vector = encoder.encode_image(str(img_path))

        payload = {
            "image_path": str(img_path.resolve()),
            "filename": img_path.name
        }

        database.upsert_vector(point_id=point_id, vector=vector, payload=payload)

        print(f"Image {img_path.name} saved with uuid: {point_id}")


if __name__ == "__main__":
    ingest_data()
