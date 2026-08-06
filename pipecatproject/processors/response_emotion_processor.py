from typing import Dict, Optional

from pipecat import Frame, FrameProcessor, PipelineContext, TextFrame


class ResponseEmotionProcessor(FrameProcessor):
    VOICE_STYLES = {
        "happy": ("energetic", "happy"),
        "sad": ("soft", "sad"),
        "angry": ("firm", "angry"),
        "fear": ("gentle", "fear"),
        "surprise": ("bright", "surprised"),
        "neutral": ("calm", "neutral"),
    }

    def __init__(self) -> None:
        super().__init__()

    async def process(self, frame: Frame, context: PipelineContext) -> Frame:
        if not isinstance(frame, TextFrame):
            return frame
        fused = frame.metadata.get("fused_emotion", {"emotion": "neutral", "confidence": 0.0})
        voice_style, tts_emotion = self.VOICE_STYLES.get(fused["emotion"], self.VOICE_STYLES["neutral"])
        frame.metadata["voice_style"] = voice_style
        frame.metadata["tts_emotion"] = tts_emotion
        return frame
