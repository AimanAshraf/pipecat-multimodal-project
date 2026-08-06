from typing import Any, Dict, Optional

from pipecat import Frame, FrameProcessor, PipelineContext
from models.emotion import EmotionLabel


class EmotionFusionProcessor(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()

    async def process(self, frame: Frame, context: PipelineContext) -> Optional[Frame]:
        face = frame.metadata.get("face_emotion", {})
        text = frame.metadata.get("text_emotion", {})
        speech = frame.metadata.get("speech_sentiment", {})

        fused = self._fuse_emotions(face, text, speech)
        frame.metadata["fused_emotion"] = fused
        context.data.setdefault("emotion_history", []).append(fused)
        return frame

    def _fuse_emotions(self, face: Dict[str, Any], text: Dict[str, Any], speech: Dict[str, Any]) -> Dict[str, Any]:
        choices = [face, text, speech]
        filtered = [item for item in choices if item and item.get("emotion")]
        if not filtered:
            return {"emotion": EmotionLabel.NEUTRAL.value, "confidence": 0.0}

        scores = {label.value: 0.0 for label in EmotionLabel}
        weights = {"face": 0.55, "text": 0.30, "speech": 0.15}

        for source, weight in zip(["face", "text", "speech"], [weights["face"], weights["text"], weights["speech"]]):
            item = locals()[source]
            emotion = item.get("emotion")
            confidence = item.get("confidence", 0.0)
            if emotion in scores:
                scores[emotion] += confidence * weight

        best_emotion = max(scores.items(), key=lambda x: x[1])[0]
        max_confidence = max(scores.values()) if scores else 0.0

        raw_confidences = []
        for item, weight in [(face, weights["face"]), (text, weights["text"]), (speech, weights["speech"])]:
            if item.get("emotion") == best_emotion:
                raw_confidences.append(item.get("confidence", 0.0))

        best_raw_confidence = max(raw_confidences) if raw_confidences else 0.0
        normalized_confidence = max(max_confidence, best_raw_confidence)
        return {"emotion": best_emotion, "confidence": float(normalized_confidence)}
