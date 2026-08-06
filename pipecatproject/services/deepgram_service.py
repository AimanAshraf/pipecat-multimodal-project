import json
import logging
from typing import Any, Dict

import httpx
from config import settings

logger = logging.getLogger(__name__)


class DeepgramService:
    base_url = "https://api.deepgram.com/v1/listen"

    def __init__(self) -> None:
        self.api_key = settings.deepgram_api_key
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/octet-stream",
        }

    async def transcribe(self, audio_bytes: bytes) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    content=audio_bytes,
                    params={"punctuate": True, "language": "en-US"},
                )
                response.raise_for_status()
                transcript_payload = response.json()
                logger.info("Deepgram transcript received")
                return transcript_payload
            except httpx.HTTPError as exc:
                logger.error("Deepgram transcription failed: %s", exc)
                raise

    def parse_transcript(self, transcript_payload: Dict[str, Any]) -> str:
        results = transcript_payload.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            return ""
        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return ""
        return alternatives[0].get("transcript", "")
