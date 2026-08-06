import json
import logging
from typing import Any, Dict

import httpx
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class CartesiaService:
    base_url = "https://api.cartesia.ai/v1"

    def __init__(self) -> None:
        self.api_key = settings.cartesia_api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def synthesize_speech(self, text: str, voice_style: str, emotion: str) -> bytes:
        url = f"{self.base_url}/tts"
        payload = {
            "text": text,
            "voice": "alloy",
            "style": voice_style,
            "emotion": emotion,
            "format": "wav",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            logger.info("Cartesia speech generated with style=%s emotion=%s", voice_style, emotion)
            return response.content
