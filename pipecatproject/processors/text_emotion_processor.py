from typing import Optional

from pipecat import Frame, FrameProcessor, PipelineContext, TextFrame
from services.emotion_service import EmotionService
from utils.logger import get_logger


logger = get_logger(__name__)


class TextEmotionProcessor(FrameProcessor):
    """
    Classifies the emotion expressed in a TextFrame.

    Normal pipeline:

        TextFrame
            ↓
        TextEmotionProcessor
            ↓
        frame.metadata["text_emotion"]
            ↓
        EmotionFusionProcessor

    The live WebSocket audio path does not use this processor for
    every audio chunk. app.py handles live speech transcription and
    speech-emotion calculation separately.
    """

    def __init__(
        self,
        emotion_service: EmotionService,
    ) -> None:
        super().__init__()

        self.emotion_service = emotion_service

        logger.info(
            "TextEmotionProcessor initialized"
        )

    async def process(
        self,
        frame: Frame,
        context: PipelineContext,
    ) -> Optional[Frame]:
        """
        Process a TextFrame and attach its emotion metadata.

        Non-text frames are passed through unchanged.
        """

        # ---------------------------------------------------------
        # Only process TextFrame objects
        # ---------------------------------------------------------

        if not isinstance(
            frame,
            TextFrame,
        ):
            return frame

        try:
            # -----------------------------------------------------
            # Get text
            # -----------------------------------------------------

            text = getattr(
                frame,
                "payload",
                "",
            )

            if text is None:
                text = ""

            text = str(text).strip()

            # -----------------------------------------------------
            # Empty text
            # -----------------------------------------------------

            if not text:
                frame.metadata[
                    "text_emotion"
                ] = {
                    "emotion": "neutral",
                    "confidence": 0.0,
                }

                logger.debug(
                    "TextEmotionProcessor received empty text"
                )

                return frame

            # -----------------------------------------------------
            # Classify emotion
            # -----------------------------------------------------

            emotion, confidence = (
                self.emotion_service.classify_text_emotion(
                    text
                )
            )

            # -----------------------------------------------------
            # Store emotion metadata
            # -----------------------------------------------------

            frame.metadata[
                "text_emotion"
            ] = {
                "emotion": emotion,
                "confidence": float(
                    confidence
                ),
            }

            # Keep the original text available to downstream
            # processors.

            frame.metadata[
                "text"
            ] = text

            logger.info(
                "Text emotion: %s %.4f | text=%s",
                emotion,
                confidence,
                text,
            )

            return frame

        except Exception as exc:
            logger.error(
                "Text emotion processing failed: %s",
                exc,
                exc_info=True,
            )

            # -----------------------------------------------------
            # Fail gracefully
            # -----------------------------------------------------

            frame.metadata[
                "text_emotion"
            ] = {
                "emotion": "neutral",
                "confidence": 0.0,
            }

            frame.metadata[
                "text_emotion_error"
            ] = str(exc)

            return frame