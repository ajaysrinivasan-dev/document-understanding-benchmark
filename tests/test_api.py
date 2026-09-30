from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_file_is_rejected():
    response = TestClient(app).post(
        "/extract", files={"file": ("notes.txt", b"text", "text/plain")}
    )
    assert response.status_code == 415
