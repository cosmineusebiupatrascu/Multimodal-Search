from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, PointIdsList
from typing import List, Dict, Any, Optional, Union

from src import config


class QdrantDB:
    def __init__(self, collection_name: str = config.COLLECTION_NAME, path: str = config.QDRANT_DB_DIR):
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = [col.name for col in self.client.get_collections().collections]

        if self.collection_name not in collections:
            print(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name = self.collection_name,
                vectors_config=VectorParams(
                    size=config.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print("Collection created")

    def get_collection_info(self):
        info = self.client.get_collection(collection_name=self.collection_name)

        return {
            "status": info.status,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count
        }

    def get_point_by_id(self, point_id: Union[int, str]):
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True
        )

        if not points:
            return None

        p = points[0]

        return {
            "id": p.id,
            "payload": p.payload,
            "vector": p.vector
        }

    def upsert_vector(self, point_id: int, vector: List[float], payload: Dict[str, Any]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

    def upsert_batch(self, points: List[Dict[str, Any]]):
        structs = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points]
        self.client.upsert(collection_name=self.collection_name, points=structs)

    def update_payload(self, point_id: Union[int, str], payload_updates: Dict[str, Any]):
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload_updates,
            points=[point_id]
        )

    def delete_points(self, point_ids: List[Union[str, int]]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=point_ids)
        )

    def clear_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)

    def search(self, query_vector: List[float], limit: int = 5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        ).points

        formatted_results = []
        for i in results:
            formatted_results.append({
                "id": i.id,
                "score": i.id,
                "payload": i.id
            })

        return formatted_results

