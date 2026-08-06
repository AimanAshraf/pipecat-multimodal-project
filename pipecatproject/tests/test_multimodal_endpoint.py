import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

import app
from pipecat import TextFrame


class DummyRunner:
    def __init__(self):
        self.tasks = {}
        self.queue = self

    async def submit(self, frame):
        task = SimpleNamespace(task_id="test-task", result=frame, error=None)
        self.tasks[task.task_id] = task
        return task

    async def join(self):
        return None


class DummyEmotionService:
    def analyze_face_emotion(self, image_bytes):
        return "happy", 0.95

    def classify_text_emotion(self, text):
        if text == "hello":
            return "happy", 0.88
        return "neutral", 0.0


@pytest.fixture(autouse=True)
def patch_app(monkeypatch):
    dummy_runner = DummyRunner()
    monkeypatch.setattr(app, "runner", dummy_runner)
    monkeypatch.setattr(app, "emotion_service", DummyEmotionService())
    monkeypatch.setattr(app.app.router, "on_startup", [])
    monkeypatch.setattr(app.app.router, "on_shutdown", [])

    async def fake_transcribe(self, audio_bytes):
        return {"results": {"channels": [{"alternatives": [{"transcript": "hello"}] }]}}

    monkeypatch.setattr(app.DeepgramService, "transcribe", fake_transcribe)

    yield


def test_multimodal_endpoint_returns_fused_metadata():
    client = TestClient(app.app)
    files = {
        "image": ("photo.jpg", b"fake-image-bytes", "image/jpeg"),
        "audio": ("voice.webm", b"fake-audio-bytes", "audio/webm"),
    }
    data = {"text": "hello"}

    response = client.post("/api/multimodal", files=files, data=data)

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript"] == "hello"
    assert payload["response"] == "hello"
    assert payload["metadata"]["face_emotion"]["emotion"] == "happy"
    assert payload["metadata"]["speech_sentiment"]["emotion"] == "happy"
    assert payload["metadata"]["text_emotion"]["emotion"] == "happy"
    assert payload["metadata"]["fused_emotion"]["emotion"] == "happy"
    assert payload["metadata"]["fused_emotion"]["confidence"] >= 0.0
