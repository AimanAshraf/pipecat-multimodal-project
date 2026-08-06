import logging
from typing import Optional

from pipecat import AudioFrame, Frame, FrameProcessor, PipelineContext, TextFrame
from services.deepgram_service import DeepgramService
from utils.logger import get_logger

logger = get_logger(__name__)


class DeepgramProcessor(FrameProcessor):
    def __init__(self, deepgram_service: Optional[DeepgramService] = None) -> None:
        super().__init__()
        self.deepgram_service = deepgram_service or DeepgramService()

    async def process(self, frame: Frame, context: PipelineContext) -> Optional[Frame]:
        if not isinstance(frame, AudioFrame):
            return frame
        logger.info("Processing audio frame through Deepgram")
        transcript_payload = await self.deepgram_service.transcribe(frame.payload)
        text = self.deepgram_service.parse_transcript(transcript_payload)
        transcript_frame = TextFrame(payload=text, metadata={"source": "deepgram"})
        transcript_frame.metadata.update(frame.metadata)
        transcript_frame.metadata["transcript"] = text
        transcript_frame.metadata["transcript_payload"] = transcript_payload
        logger.debug("Deepgram transcript frame created")
        return transcript_frame
