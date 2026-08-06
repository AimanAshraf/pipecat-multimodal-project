from datetime import datetime
from typing import Optional

from pipecat import Frame, FrameProcessor, PipelineContext, TextFrame


class ConversationContextProcessor(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()

    async def process(self, frame: Frame, context: PipelineContext) -> Optional[Frame]:
        if not isinstance(frame, TextFrame):
            return frame
        history = context.data.setdefault("conversation_history", [])
        user_state = {
            "role": "user",
            "text": frame.payload,
            "timestamp": datetime.utcnow().isoformat(),
            "emotion": frame.metadata.get("fused_emotion", {}),
        }
        history.append(user_state)
        context.data["last_user_message"] = frame.payload
        return frame
