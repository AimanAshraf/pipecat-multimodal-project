import logging
from typing import Optional

from pipecat import Frame, FrameProcessor, ImageFrame, PipelineContext, TextFrame
from services.emotion_service import EmotionService
from utils.logger import get_logger

logger = get_logger(__name__)


class FaceEmotionProcessor(FrameProcessor):
    def __init__(self, emotion_service: Optional[EmotionService] = None) -> None:
        super().__init__()
        self.emotion_service = emotion_service or EmotionService(model_name="j-hartmann/emotion-english-distilroberta-base")

    async def process(self, frame: Frame, context: PipelineContext) -> Optional[Frame]:
        if not isinstance(frame, ImageFrame):
            return frame
        logger.info("Extracting facial emotion from image frame")
        emotion_label, confidence = self.emotion_service.analyze_face_emotion(frame.payload)
        frame.metadata["face_emotion"] = {
            "emotion": emotion_label,
            "confidence": float(confidence),
        }
        logger.debug("Face emotion metadata set: %s", frame.metadata["face_emotion"])
        return frame
