import time
import uuid
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from qdrant_client.models import PointStruct

from src.search import SearchEngine
from src.vector_db import QdrantDB

IMAGE_DIR = Path("data/raw_images")
BATCH_SIZE = 32


def main():
    search_engine = SearchEngine()
    database = QdrantDB()

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in valid_extensions]

    total_images = len(image_paths)
    print(f"{total_images} images found")

    if total_images == 0:
        return

    start_time = time.time()

    for i in tqdm(range(0, total_images, BATCH_SIZE)):
        batch_paths = image_paths[i: i + BATCH_SIZE]
        points = []

        for file_path in batch_paths:
            try:
                image = Image.open(file_path).convert("RGB")

                vector = search_engine.encoder.encode_image(image)

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_path.name))
                image_url = f"/static/{file_path.name}"

                payload = {
                    "image_url": image_url,
                    "filename": file_path.name
                }

                # PointStruct este formatul nativ pentru batch-uri Qdrant
                points.append(
                    PointStruct(id=point_id, vector=vector, payload=payload)
                )
            except Exception as e:
                print(f"Eroare la procesarea fișierului {file_path.name}: {e}")

        if points:
            database.client.upsert(
                collection_name=database.collection_name,
                points=points
            )

        processed_count = min(i + BATCH_SIZE, total_images)
        print(f"Progress: {processed_count}/{total_images} images indexed")

    elapsed = time.time() - start_time
    print(f"Process finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()