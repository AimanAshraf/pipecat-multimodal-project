import logging
from typing import Optional

from pipecat import Frame, FrameProcessor, PipelineContext, TextFrame
from services.emotion_service import EmotionService
from utils.logger import get_logger

logger = get_logger(__name__)


class TextEmotionProcessor(FrameProcessor):
    def __init__(self, emotion_service: Optional[EmotionService] = None) -> None:
        super().__init__()
        self.emotion_service = emotion_service or EmotionService(model_name="j-hartmann/emotion-english-distilroberta-base")

    async def process(self, frame: Frame, context: PipelineContext) -> Optional[Frame]:
        if not isinstance(frame, TextFrame):
            return frame
        logger.info("Predicting text emotion")
        emotion_label, confidence = self.emotion_service.classify_text_emotion(frame.payload)
        frame.metadata["text_emotion"] = {
            "emotion": emotion_label,
            "confidence": float(confidence),
        }
        logger.debug("Text emotion metadata set: %s", frame.metadata["text_emotion"])
        return frame
