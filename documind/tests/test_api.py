from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "DocuMind"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "gemini_key_configured" in response.json()

def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
