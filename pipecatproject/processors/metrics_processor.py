import time

from pipecat import Frame, FrameProcessor, PipelineContext


class MetricsProcessor(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()

    async def process(self, frame: Frame, context: PipelineContext) -> Frame:
        started_at = time.monotonic()
        result = frame
        result.metadata["metrics"] = {
            "received_at": started_at,
        }
        return result
