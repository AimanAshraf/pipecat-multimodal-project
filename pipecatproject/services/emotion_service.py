import logging
from typing import Dict, Tuple

import cv2
import numpy as np
from fer.fer import FER
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from models.emotion import EmotionLabel
from utils.logger import get_logger

logger = get_logger(__name__)


class EmotionService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )
        self.face_detector = FER(mtcnn=True)

    def classify_text_emotion(self, text: str) -> Tuple[str, float]:
        if not text.strip():
            return EmotionLabel.NEUTRAL.value, 0.0

        try:
            results = self.pipeline(text)
            if not results:
                return EmotionLabel.NEUTRAL.value, 0.0

            if isinstance(results, dict):
                results = [results]

            if isinstance(results[0], dict) and "label" in results[0]:
                scores = {
                    self._normalize_label(item["label"]): item.get("score", 0.0)
                    for item in results
                    if isinstance(item, dict) and "label" in item
                }
            elif isinstance(results[0], list):
                scores = {
                    self._normalize_label(item["label"]): item.get("score", 0.0)
                    for item in results[0]
                    if isinstance(item, dict) and "label" in item
                }
            else:
                return EmotionLabel.NEUTRAL.value, 0.0

            if not scores:
                return EmotionLabel.NEUTRAL.value, 0.0

            mapped = self._map_label(scores)
            logger.info("Text emotion classified: %s %s", mapped[0], mapped[1])
            return mapped
        except Exception as exc:
            logger.error("Text emotion classification failed: %s", exc, exc_info=True)
            return EmotionLabel.NEUTRAL.value, 0.0

    def analyze_face_emotion(self, image_bytes: bytes) -> Tuple[str, float]:
        try:
            image = self._decode_image(image_bytes)
            logger.info("Decoded image shape: %s", image.shape)
            detection = self.face_detector.detect_emotions(image)
            logger.info("FER raw detection result: %s", detection)
            if detection:
                top_face = detection[0]
                emotions = top_face.get("emotions", {})
                logger.info("Raw emotion scores: %s", emotions)
                mapped = self._map_label(emotions)
                if mapped[0] != EmotionLabel.NEUTRAL.value or max(emotions.values(), default=0.0) > 0.3:
                    return mapped
                logger.info("FER mapped emotion was neutral or low confidence, falling back to top_emotion")

            top_label, top_score = self.face_detector.top_emotion(image)
            logger.info("FER top emotion fallback: %s %s", top_label, top_score)
            if top_label:
                return self._map_label({top_label: top_score})

            logger.warning("No face detected in frame")
            return EmotionLabel.NEUTRAL.value, 0.0
        except Exception as exc:
            logger.error("Face emotion analysis failed: %s", exc, exc_info=True)
            return EmotionLabel.NEUTRAL.value, 0.0

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes into an RGB numpy array for FER/MTCNN."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode image bytes; not a valid image format")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def _normalize_label(self, label: str) -> str:
        normalized_label = label.strip().lower()
        label_map = {
            "joy": "happy",
            "happiness": "happy",
            "sadness": "sad",
            "anger": "angry",
            "fear": "fear",
            "surprise": "surprise",
            "neutral": "neutral",
            "calm": "neutral",
            "disgust": "angry",
            "contentment": "happy",
            "trust": "happy",
            "anticipation": "neutral",
        }
        return label_map.get(normalized_label, normalized_label)

    def _map_label(self, scores: Dict[str, float]) -> Tuple[str, float]:
        normalized = {key.lower(): value for key, value in scores.items()}
        candidates = {
            EmotionLabel.HAPPY.value: normalized.get("happy", 0.0),
            EmotionLabel.SAD.value: normalized.get("sad", 0.0),
            EmotionLabel.ANGRY.value: normalized.get("angry", 0.0),
            EmotionLabel.FEAR.value: normalized.get("fear", 0.0),
            EmotionLabel.SURPRISE.value: normalized.get("surprise", 0.0),
            EmotionLabel.NEUTRAL.value: normalized.get("neutral", 0.0),
        }
        best = max(candidates.items(), key=lambda item: item[1])
        return best