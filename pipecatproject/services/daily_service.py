import logging
from typing import Dict

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class DailyService:
    def __init__(self) -> None:
        self.api_key = settings.daily_api_key
        self.domain = settings.daily_domain

    def room_url(self) -> str:
        return f"https://{self.domain}/{settings.daily_room_name}"

    def call_token_payload(self) -> Dict[str, str]:
        return {
            "properties": {
                "enable_screenshare": False,
                "enable_chat": True,
                "enable_network_ui": True,
            }
        }
