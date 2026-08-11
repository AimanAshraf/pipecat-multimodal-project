import asyncio
import json
import logging
from typing import Any, Dict, Optional

import httpx
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqService:
    base_url = "https://api.groq.com/openai/v1"

    def __init__(self) -> None:
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def stream_completion(self, prompt: str) -> Any:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=self.headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    stripped = line.strip()
                    if stripped == "data: [DONE]":
                        continue

                    if stripped.startswith("data:"):
                        payload_text = stripped[len("data:"):].strip()
                    else:
                        payload_text = stripped

                    if not payload_text:
                        continue

                    try:
                        part = json.loads(payload_text)
                        yield part
                    except json.JSONDecodeError:
                        continue

    async def generate_response(self, prompt: str) -> str:
        response_text = []
        try:
            async for part in self.stream_completion(prompt):
                if not isinstance(part, dict):
                    logger.warning("Groq stream yielded non-dict part: %r", part)
                    continue
                for choice in part.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if content:
                        response_text.append(content)
        except Exception as exc:
            logger.error("Groq response generation failed: %s", exc, exc_info=True)
            return "Sorry, I couldn't generate a response at the moment."

        result = "".join(response_text).strip()
        if not result:
            logger.warning("Groq generated an empty response for prompt")
            return "Sorry, I couldn't generate a response at the moment."
        return result

    async def generate(self, prompt: str) -> str:
        """Alias for generate_response"""
        return await self.generate_response(prompt)
