import io
import pytest
from unittest.mock import patch, ANY
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


def generate_dummy_image():
    file = io.BytesIO()
    image = Image.new("RGB", (10, 10), color="white")
    image.save(file, "jpeg")
    file.seek(0)
    return file


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "service": "multimodal-search-api"
        }


@patch("src.api.main.SearchEngine")
def test_search_by_text_success(MockSearchEngine):
    mock_engine = MockSearchEngine.return_value
    mock_engine.search_by_text.return_value = [
        {"id": "uuid-1", "score": 0.95, "payload": {"filename": "result.jpg"}}
    ]

    with TestClient(app) as client:
        response = client.post("/search/text", data={"query": "test query", "limit": 3})

        assert response.status_code == 200
        assert response.json()[0]["id"] == "uuid-1"
        mock_engine.search_by_text.assert_called_once_with(query_text="test query", limit=3)


def test_search_by_text_empty_query():
    with TestClient(app) as client:
        response = client.post("/search/text", data={"query": "   ", "limit": 5})

        assert response.status_code == 400
        assert response.json()["detail"] == "Query text cannot be empty"


@patch("src.api.main.SearchEngine")
def test_search_by_image_success(MockSearchEngine):
    mock_engine = MockSearchEngine.return_value
    mock_engine.search_by_image.return_value = [
        {"id": "uuid-2", "score": 0.88, "payload": {"filename": "match.jpg"}}
    ]

    with TestClient(app) as client:
        dummy_image = generate_dummy_image()

        response = client.post(
            "/search/image",
            data={"limit": 5},
            files={"file": ("test.jpg", dummy_image, "image/jpeg")}
        )

        assert response.status_code == 200
        assert response.json()[0]["id"] == "uuid-2"
        mock_engine.search_by_image.assert_called_once_with(query_image=ANY, limit=5)


def test_search_by_image_invalid_type():
    with TestClient(app) as client:
        response = client.post(
            "/search/image",
            data={"limit": 5},
            files={"file": ("document.txt", b"dummy string data", "text/plain")}
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "File must be an image"


@patch("src.api.main.QdrantDB")
def test_delete_points_success(MockQdrantDB):
    mock_db_instance = MockQdrantDB.return_value
    mock_db_instance.delete_points.return_value = None

    with TestClient(app) as client:
        payload = {"point_ids": ["uuid-1", "uuid-2"]}
        response = client.post("/delete/points", json=payload)

        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "deleted_count": 2
        }

        mock_db_instance.delete_points.assert_called_once_with(point_ids=["uuid-1", "uuid-2"])


def test_delete_points_validation_empty():
    with TestClient(app) as client:
        response = client.post("/delete/points", json={"point_ids": []})
        assert response.status_code == 400
        assert response.json()["detail"] == "IDs list cannot be empty"


def test_delete_points_validation_invalid_type():
    with TestClient(app) as client:
        response = client.post("/delete/points", json={"point_ids": "not_a_list"})
        assert response.status_code == 422