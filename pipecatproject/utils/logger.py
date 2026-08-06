import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

handler = RotatingFileHandler(LOG_DIR / "assistant.log", maxBytes=5_242_880, backupCount=3)
handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console_handler)
    return logger
