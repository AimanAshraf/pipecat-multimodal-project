import logging
from pipecat import Frame, FrameProcessor, PipelineContext
from utils.logger import get_logger

logger = get_logger(__name__)


class LoggingProcessor(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()

    async def process(self, frame: Frame, context: PipelineContext) -> Frame:
        logger.info(
            "Frame %s processed: type=%s metadata=%s",
            frame.frame_id,
            type(frame).__name__,
            {k: v for k, v in frame.metadata.items() if k in ["face_emotion", "text_emotion", "fused_emotion", "voice_style"]},
        )
        return frame
