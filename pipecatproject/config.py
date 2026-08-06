import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    cartesia_api_key: str = os.getenv("CARTESIA_API_KEY", "")
    daily_api_key: str = os.getenv("DAILY_API_KEY", "")
    daily_domain: str = os.getenv("DAILY_DOMAIN", "")
    huggingface_model: str = os.getenv("HUGGINGFACE_MODEL", "j-hartmann/emotion-english-distilroberta-base")
    daily_room_name: str = os.getenv("DAILY_ROOM_NAME", "emotion-assistant-room")
    app_name: str = os.getenv("APP_NAME", "Emotion-Aware Conversational AI Assistant")
    enable_debug: bool = os.getenv("ENABLE_DEBUG", "false").lower() in ("1", "true", "yes")


settings = Settings()
