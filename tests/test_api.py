\"\"\"Basic FastAPI smoke test.\"\"\"
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz() -> None:
    r = client.get(\"/healthz\")
    assert r.status_code == 200
    assert r.json() == {\"status\": \"ok\"}
