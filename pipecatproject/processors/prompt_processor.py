from typing import Dict, List

from pipecat import Frame, FrameProcessor, PipelineContext, TextFrame
from services.groq_service import GroqService


class PromptProcessor(FrameProcessor):
    def __init__(self, groq_service: GroqService) -> None:
        super().__init__()
        self.groq_service = groq_service

    async def process(self, frame: Frame, context: PipelineContext) -> Frame:
        if not isinstance(frame, TextFrame):
            return frame
        prompt = self._build_prompt(frame, context.data)
        frame.metadata["generated_prompt"] = prompt
        response_text = await self.groq_service.generate_response(prompt)
        response_frame = TextFrame(payload=response_text, metadata={"source": "groq"})
        response_frame.metadata.update(frame.metadata)
        # Also store the generated response in metadata so callers can access it
        response_frame.metadata["response_text"] = response_text
        return response_frame

    def _build_prompt(self, frame: TextFrame, context_data: Dict[str, List[Dict[str, str]]]) -> str:
        fused = frame.metadata.get("fused_emotion", {"emotion": "neutral", "confidence": 0.0})
        history = context_data.get("conversation_history", [])[-5:]
        history_lines = []
        for event in history:
            history_lines.append(f"{event['role'].title()}: {event['text']} (Emotion: {event['emotion'].get('emotion')} Confidence: {event['emotion'].get('confidence')})")
        history_text = "\n".join(history_lines)
        prompt = (
            "You are an empathetic assistant that responds to a user with emotional awareness. "
            "Do not hallucinate feelings. Always acknowledge the user emotion. Keep the response concise and supportive.\n\n"
            f"User Emotion: {fused['emotion'].title()}\n"
            f"Confidence: {fused['confidence']:.2f}\n\n"
            "Conversation History:\n"
            f"{history_text}\n\n"
            "Current User Message:\n"
            f"\"{frame.payload}\"\n\n"
            "Please respond empathetically with a calm tone, while mentioning the emotional state, and avoid making assumptions.")
        return prompt
