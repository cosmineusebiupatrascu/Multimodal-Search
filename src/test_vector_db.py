from src.encoder import CLIPEncoder
from src.vector_db import QdrantDB

encoder = CLIPEncoder()
database = QdrantDB()

test_text = "A photo of a red sports car"
vector = encoder.encode_text(test_text)

database.upsert_vector(
    point_id=1,
    vector=vector,
    payload={"image_path": "data/raw_images/car/jpg", "description": test_text}
)

search_vector = encoder.encode_text("red_car")
results = database.search(query_vector=search_vector, limit=1)

print(results)
