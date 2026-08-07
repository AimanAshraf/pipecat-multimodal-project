import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from models.emotion import EmotionLabel
from utils.logger import get_logger

logger = get_logger(__name__)

# Torch-native facial emotion classifier — replaces fer/keras/tensorflow.
FACE_EMOTION_MODEL = "trpakov/vit-face-expression"


class EmotionService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._text_pipeline = None
        self._face_detector: Optional[MTCNN] = None
        self._face_emotion_pipeline = None

    def _quantize(self, model: torch.nn.Module, label: str) -> torch.nn.Module:
        """Apply int8 dynamic quantization to linear layers to reduce memory footprint."""
        try:
            quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("%s model quantized to int8", label)
            return quantized
        except Exception as exc:
            logger.warning("Quantization failed for %s, falling back to fp32: %s", label, exc)
            return model

    def _ensure_text_pipeline(self) -> None:
        """Lazily instantiate the text-emotion model on first use."""
        if self._text_pipeline is None:
            logger.info("Loading text emotion model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._model = self._quantize(self._model, "Text emotion")
            self._text_pipeline = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                return_all_scores=True,
            )

    def _ensure_face_pipeline(self) -> None:
        """Lazily instantiate face detector + emotion classifier on first use."""
        if self._face_detector is None:
            logger.info("Loading MTCNN face detector")
            self._face_detector = MTCNN(keep_all=False, post_process=False, device="cpu")
        if self._face_emotion_pipeline is None:
            logger.info("Loading face emotion model: %s", FACE_EMOTION_MODEL)
            self._face_emotion_pipeline = pipeline(
                "image-classification",
                model=FACE_EMOTION_MODEL,
                device=-1,  # CPU
            )
            self._face_emotion_pipeline.model.eval()
            self._face_emotion_pipeline.model = self._quantize(
                self._face_emotion_pipeline.model, "Face emotion"
            )

    def classify_text_emotion(self, text: str) -> Tuple[str, float]:
        if not text.strip():
            return EmotionLabel.NEUTRAL.value, 0.0

        try:
            self._ensure_text_pipeline()
            results = self._text_pipeline(text)
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
            self._ensure_face_pipeline()
            image = self._decode_image(image_bytes)
            logger.info("Decoded image shape: %s", image.shape)

            pil_image = Image.fromarray(image)
            boxes, probs = self._face_detector.detect(pil_image)

            if boxes is None or len(boxes) == 0:
                logger.warning("No face detected in frame")
                return EmotionLabel.NEUTRAL.value, 0.0

            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
            face_crop = pil_image.crop((x1, y1, x2, y2))

            if face_crop.width == 0 or face_crop.height == 0:
                logger.warning("Detected face crop was empty")
                return EmotionLabel.NEUTRAL.value, 0.0

            results = self._face_emotion_pipeline(face_crop)
            logger.info("Face emotion raw results: %s", results)

            scores = {
                self._normalize_label(item["label"]): item["score"]
                for item in results
                if "label" in item and "score" in item
            }
            if not scores:
                return EmotionLabel.NEUTRAL.value, 0.0

            mapped = self._map_label(scores)
            logger.info("Face emotion classified: %s %s", mapped[0], mapped[1])
            return mapped
        except Exception as exc:
            logger.error("Face emotion analysis failed: %s", exc, exc_info=True)
            return EmotionLabel.NEUTRAL.value, 0.0

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes into an RGB numpy array."""
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