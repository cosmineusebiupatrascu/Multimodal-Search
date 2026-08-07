from PIL import Image
from typing import List, Dict, Any, Union

from src.encoder import CLIPEncoder
from src.vector_db import QdrantDB


class SearchEngine:
    def __init__(self):
        self.encoder = CLIPEncoder()
        self.database = QdrantDB()

    def search_by_text(self, query_text: str, limit: int = 5):
        query_vector = self.encoder.encode_text(query_text)
        return self.database.search(query_vector=query_vector, limit=limit)

    def search_by_image(self, query_image: Union[str, Image.Image], limit: int = 5):
        query_vector = self.encoder.encode_image(query_image)
        return self.database.search(query_vector=query_vector, limit=limit)