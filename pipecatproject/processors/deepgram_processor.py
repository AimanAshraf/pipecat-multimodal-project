from typing import Optional

from pipecat import Frame, FrameProcessor, PipelineContext, TextFrame
from services.deepgram_service import DeepgramService
from utils.logger import get_logger


logger = get_logger(__name__)


class DeepgramProcessor(FrameProcessor):
    """
    Converts AudioFrame data into a TextFrame using Deepgram.

    This processor is used by the normal HTTP/Pipecat pipeline.

    The continuous live WebSocket audio path in app.py does NOT use
    this processor for every 750 ms chunk. It communicates directly
    with DeepgramService so that transcripts can be returned to the
    browser immediately.

    Normal pipeline:

        AudioFrame
            ↓
        DeepgramProcessor
            ↓
        TextFrame
            ↓
        TextEmotionProcessor
            ↓
        EmotionFusionProcessor
            ↓
        ConversationContextProcessor
            ↓
        PromptProcessor
            ↓
        ResponseEmotionProcessor
    """

    def __init__(
        self,
        deepgram_service: Optional[DeepgramService] = None,
    ) -> None:
        super().__init__()

        self.deepgram_service = (
            deepgram_service
            if deepgram_service is not None
            else DeepgramService()
        )

        logger.info(
            "DeepgramProcessor initialized"
        )

    async def process(
        self,
        frame: Frame,
        context: PipelineContext,
    ) -> Optional[Frame]:
        """
        Process an incoming frame.

        AudioFrame:
            Send audio bytes to Deepgram and return TextFrame.

        Other frames:
            Pass through unchanged.
        """

        # ---------------------------------------------------------
        # Only process audio frames
        # ---------------------------------------------------------

        if frame.__class__.__name__ != "AudioFrame":
            return frame

        audio_payload = getattr(
            frame,
            "payload",
            b"",
        )

        if not audio_payload:
            logger.warning(
                "DeepgramProcessor received empty audio payload"
            )

            return TextFrame(
                payload="",
                metadata={
                    "transcript": "",
                    "source": "deepgram",
                },
            )

        try:
            # -----------------------------------------------------
            # Send audio to Deepgram
            # -----------------------------------------------------

            result = await self.deepgram_service.transcribe(
                audio_payload
            )

            # -----------------------------------------------------
            # Parse Deepgram response
            # -----------------------------------------------------

            transcript = (
                self.deepgram_service.parse_transcript(
                    result
                )
            )

            if transcript is None:
                transcript = ""

            transcript = str(
                transcript
            ).strip()

            logger.info(
                "Deepgram transcript: %s",
                transcript,
            )

            # -----------------------------------------------------
            # Preserve useful metadata
            # -----------------------------------------------------

            metadata = dict(
                getattr(
                    frame,
                    "metadata",
                    {},
                )
                or {}
            )

            metadata.update(
                {
                    "transcript": transcript,
                    "source": "deepgram",
                }
            )

            # -----------------------------------------------------
            # Return TextFrame
            # -----------------------------------------------------

            return TextFrame(
                payload=transcript,
                metadata=metadata,
            )

        except Exception as exc:
            logger.error(
                "Deepgram transcription failed: %s",
                exc,
                exc_info=True,
            )

            # Do not crash the entire pipeline because of a
            # temporary transcription failure.

            metadata = dict(
                getattr(
                    frame,
                    "metadata",
                    {},
                )
                or {}
            )

            metadata.update(
                {
                    "transcript": "",
                    "source": "deepgram",
                    "transcription_error": str(exc),
                }
            )

            return TextFrame(
                payload="",
                metadata=metadata,
            )