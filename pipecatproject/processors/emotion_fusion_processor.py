from typing import Dict, Optional, Tuple

from pipecat import Frame, FrameProcessor, PipelineContext

from utils.logger import get_logger


logger = get_logger(__name__)


class EmotionFusionProcessor(FrameProcessor):
    """
    Weighted multimodal emotion fusion.

    Normal pipeline weights:

        face   = 0.55
        text   = 0.30
        speech = 0.15

    The live WebSocket fusion is handled separately by app.py.
    """

    FACE_WEIGHT = 0.55
    TEXT_WEIGHT = 0.30
    SPEECH_WEIGHT = 0.15

    EMOTIONS = (
        "happy",
        "sad",
        "angry",
        "fear",
        "surprise",
        "neutral",
    )

    def __init__(self) -> None:
        super().__init__()

        logger.info(
            "EmotionFusionProcessor initialized "
            "face=%.2f text=%.2f speech=%.2f",
            self.FACE_WEIGHT,
            self.TEXT_WEIGHT,
            self.SPEECH_WEIGHT,
        )

    async def process(
        self,
        frame: Frame,
        context: PipelineContext,
    ) -> Optional[Frame]:

        metadata = getattr(
            frame,
            "metadata",
            None,
        )

        if metadata is None:
            metadata = {}
            frame.metadata = metadata

        try:
            face = self._extract_emotion(
                metadata.get("face_emotion")
            )

            text = self._extract_emotion(
                metadata.get("text_emotion")
            )

            speech = self._extract_emotion(
                metadata.get("speech_sentiment")
            )

            scores = self._fuse(
                face=face,
                text=text,
                speech=speech,
            )

            emotion, confidence = (
                self._get_best_emotion(scores)
            )

            result = {
                "emotion": emotion,
                "confidence": round(
                    confidence,
                    4,
                ),
                "scores": {
                    key: round(
                        value,
                        4,
                    )
                    for key, value in scores.items()
                },
            }

            metadata["fused_emotion"] = result
            metadata["dominant_emotion"] = emotion
            metadata["emotion_confidence"] = round(
                confidence,
                4,
            )

            logger.info(
                "Fused emotion: %s %.4f",
                emotion,
                confidence,
            )

            return frame

        except Exception as exc:

            logger.error(
                "Emotion fusion failed: %s",
                exc,
                exc_info=True,
            )

            fallback = {
                "emotion": "neutral",
                "confidence": 0.0,
                "scores": self._empty_scores(),
            }

            metadata["fused_emotion"] = fallback
            metadata["dominant_emotion"] = "neutral"
            metadata["emotion_confidence"] = 0.0

            return frame

    # ============================================================
    # EXTRACT
    # ============================================================

    def _extract_emotion(
        self,
        value,
    ) -> Tuple[str, float]:

        if value is None:
            return "neutral", 0.0

        if isinstance(value, dict):

            emotion = value.get(
                "emotion",
                value.get(
                    "label",
                    "neutral",
                ),
            )

            confidence = value.get(
                "confidence",
                value.get(
                    "score",
                    0.0,
                ),
            )

        elif isinstance(
            value,
            (tuple, list),
        ):

            if len(value) < 2:
                return "neutral", 0.0

            emotion = value[0]
            confidence = value[1]

        else:
            return "neutral", 0.0

        emotion = self._normalize_emotion(
            str(emotion)
        )

        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        if emotion not in self.EMOTIONS:
            return "neutral", 0.0

        return emotion, confidence

    # ============================================================
    # FUSION
    # ============================================================

    def _fuse(
        self,
        face: Tuple[str, float],
        text: Tuple[str, float],
        speech: Tuple[str, float],
    ) -> Dict[str, float]:

        scores = self._empty_scores()

        modalities = (
            (face, self.FACE_WEIGHT),
            (text, self.TEXT_WEIGHT),
            (speech, self.SPEECH_WEIGHT),
        )

        for emotion_data, weight in modalities:

            emotion, confidence = emotion_data

            if emotion not in scores:
                continue

            scores[emotion] += (
                confidence * weight
            )

        return scores

    # ============================================================
    # BEST EMOTION
    # ============================================================

    def _get_best_emotion(
        self,
        scores: Dict[str, float],
    ) -> Tuple[str, float]:

        if not scores:
            return "neutral", 0.0

        max_score = max(
            scores.values()
        )

        # IMPORTANT:
        # Do not return "happy" merely because it is
        # the first dictionary key when there is no evidence.

        if max_score <= 0.0:
            return "neutral", 0.0

        emotion = max(
            scores,
            key=scores.get,
        )

        return emotion, float(
            scores[emotion]
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def _normalize_emotion(
        self,
        emotion: str,
    ) -> str:

        normalized = (
            emotion
            .strip()
            .lower()
        )

        aliases = {
            "joy": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry",
            "calm": "neutral",
            "contentment": "happy",
            "trust": "happy",
            "disgust": "angry",
            "anticipation": "neutral",
        }

        return aliases.get(
            normalized,
            normalized,
        )

    # ============================================================
    # EMPTY SCORES
    # ============================================================

    def _empty_scores(
        self,
    ) -> Dict[str, float]:

        return {
            emotion: 0.0
            for emotion in self.EMOTIONS
        }