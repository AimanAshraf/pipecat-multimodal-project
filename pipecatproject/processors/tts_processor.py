from typing import Optional

from pipecat import AudioFrame, Frame, FrameProcessor, PipelineContext, TextFrame
from services.cartesia_service import CartesiaService


class TTSProcessor(FrameProcessor):
    def __init__(self, cartesia_service: Optional[CartesiaService] = None) -> None:
        super().__init__()
        self.cartesia_service = cartesia_service or CartesiaService()

    async def process(self, frame: Frame, context: PipelineContext) -> Frame:
        if not isinstance(frame, TextFrame):
            return frame
        voice_style = frame.metadata.get("voice_style", "calm")
        tts_emotion = frame.metadata.get("tts_emotion", "neutral")
        audio_bytes = await self.cartesia_service.synthesize_speech(frame.payload, voice_style, tts_emotion)
        audio_frame = AudioFrame(payload=audio_bytes, metadata=frame.metadata.copy())
        audio_frame.metadata["tts_generated"] = True
        return audio_frame
