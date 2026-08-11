import logging
from typing import Dict, Tuple, Optional
import os

import cv2
import numpy as np

# Fix for Python 3.13 compatibility with FER
try:
    import pkg_resources
except ModuleNotFoundError:
    # Python 3.13+ deprecated pkg_resources
    # Create a functional shim for FER compatibility
    import sys
    from types import ModuleType
    from pathlib import Path
    
    pkg_resources = ModuleType('pkg_resources')
    
    def resource_filename(package_name, resource_name):
        """Minimal implementation for FER model loading"""
        # FER tries to load its model from its package directory
        try:
            import fer
            fer_path = Path(fer.__file__).parent
            return str(fer_path / resource_name)
        except Exception:
            # Fallback: return the resource name as-is
            return resource_name
    
    pkg_resources.resource_filename = resource_filename
    sys.modules['pkg_resources'] = pkg_resources

from fer.fer import FER
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from models.emotion import EmotionLabel
from utils.logger import get_logger


logger = get_logger(__name__)


class EmotionService:
    """
    Centralized emotion-analysis service.

    Supports:
        1. Text emotion classification
        2. Face emotion classification from webcam frames
        3. Speech emotion classification from transcribed audio
        4. Normalization of all emotion labels into the application's
           six supported emotions.

    Supported emotions:
        - happy
        - sad
        - angry
        - fear
        - surprise
        - neutral

    The actual multimodal fusion is intentionally kept outside this
    service in EmotionFusionProcessor.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        logger.info(
            "Loading text emotion model: %s",
            model_name,
        )

        # ---------------------------------------------------------
        # TEXT EMOTION MODEL
        # ---------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )

        # ---------------------------------------------------------
        # FACE EMOTION MODEL
        # ---------------------------------------------------------

        logger.info("Loading FER face emotion detector...")

        self.face_detector = FER(
            mtcnn=True
        )

        logger.info("EmotionService initialized successfully.")

    # =============================================================
    # TEXT EMOTION
    # =============================================================

    def classify_text_emotion(
        self,
        text: str,
    ) -> Tuple[str, float]:
        """
        Classify emotion from text.

        Returns:
            (emotion, confidence)
        """

        if not text or not text.strip():
            return (
                EmotionLabel.NEUTRAL.value,
                0.0,
            )

        try:
            results = self.pipeline(text)

            if not results:
                return (
                    EmotionLabel.NEUTRAL.value,
                    0.0,
                )

            # Depending on the Transformers version/configuration,
            # pipeline output can be:
            #
            # [
            #     {"label": "...", "score": ...},
            #     ...
            # ]
            #
            # or:
            #
            # [
            #     [
            #         {"label": "...", "score": ...},
            #         ...
            #     ]
            # ]

            if isinstance(results, dict):
                results = [results]

            if (
                isinstance(results, list)
                and results
                and isinstance(results[0], list)
            ):
                results = results[0]

            if not isinstance(results, list):
                return (
                    EmotionLabel.NEUTRAL.value,
                    0.0,
                )

            scores: Dict[str, float] = {}

            for item in results:
                if not isinstance(item, dict):
                    continue

                label = item.get("label")

                if not label:
                    continue

                score = float(
                    item.get("score", 0.0)
                )

                normalized_label = self._normalize_label(
                    label
                )

                # Multiple source labels can map to the same
                # application emotion. Keep the highest score.
                scores[normalized_label] = max(
                    scores.get(normalized_label, 0.0),
                    score,
                )

            if not scores:
                return (
                    EmotionLabel.NEUTRAL.value,
                    0.0,
                )

            result = self._map_label(scores)

            logger.info(
                "Text emotion: %s (%.4f)",
                result[0],
                result[1],
            )

            return result

        except Exception as exc:
            logger.error(
                "Text emotion classification failed: %s",
                exc,
                exc_info=True,
            )

            return (
                EmotionLabel.NEUTRAL.value,
                0.0,
            )

    # =============================================================
    # SPEECH EMOTION
    # =============================================================

    def classify_speech_emotion(
        self,
        transcript: str,
    ) -> Tuple[str, float]:
        """
        Classify emotion from speech transcription.

        The current architecture uses the speech transcript as the
        semantic representation of the user's voice.

        Deepgram is responsible for STT.
        This method is responsible for emotion classification.

        This is intentionally separate from text classification so
        the WebSocket/audio pipeline can clearly distinguish:

            microphone
                ↓
            Deepgram STT
                ↓
            speech emotion
        """

        if not transcript or not transcript.strip():
            return (
                EmotionLabel.NEUTRAL.value,
                0.0,
            )

        return self.classify_text_emotion(
            transcript
        )

    # =============================================================
    # FACE EMOTION
    # =============================================================

    def analyze_face_emotion(
        self,
        image_bytes: bytes,
    ) -> Tuple[str, float]:
        """
        Analyze facial emotion from a JPEG/PNG image.

        The image originates from the browser webcam.

        Important:
            There is NO cv2.VideoCapture(0) here.

        This allows the backend to run on AWS/cloud infrastructure
        without requiring access to a server-side webcam.

        Returns:
            (emotion, confidence)
        """

        try:
            image = self._decode_image(
                image_bytes
            )

            logger.debug(
                "Decoded image shape: %s",
                image.shape,
            )

            detection = (
                self.face_detector.detect_emotions(
                    image
                )
            )

            logger.debug(
                "FER detection count: %s",
                len(detection) if detection else 0,
            )

            # -----------------------------------------------------
            # FACE FOUND
            # -----------------------------------------------------

            if detection:
                # FER can detect multiple faces.
                #
                # For the current application we use the face with
                # the largest bounding box, rather than blindly
                # assuming detection[0] is always the user.
                top_face = self._select_primary_face(
                    detection
                )

                if top_face:
                    emotions = top_face.get(
                        "emotions",
                        {},
                    )

                    if emotions:
                        logger.debug(
                            "Face emotion scores: %s",
                            emotions,
                        )

                        emotion, confidence = (
                            self._map_label(
                                emotions
                            )
                        )

                        logger.info(
                            "Face emotion: %s (%.4f)",
                            emotion,
                            confidence,
                        )

                        return (
                            emotion,
                            confidence,
                        )

            # -----------------------------------------------------
            # FALLBACK
            # -----------------------------------------------------

            logger.debug(
                "No usable FER detection; trying top_emotion()."
            )

            top_label, top_score = (
                self.face_detector.top_emotion(
                    image
                )
            )

            if top_label:
                normalized = self._normalize_label(
                    top_label
                )

                logger.info(
                    "FER fallback emotion: %s (%.4f)",
                    normalized,
                    float(top_score or 0.0),
                )

                return (
                    normalized,
                    float(top_score or 0.0),
                )

            logger.debug(
                "No face detected."
            )

            return (
                EmotionLabel.NEUTRAL.value,
                0.0,
            )

        except Exception as exc:
            logger.error(
                "Face emotion analysis failed: %s",
                exc,
                exc_info=True,
            )

            return (
                EmotionLabel.NEUTRAL.value,
                0.0,
            )

    # =============================================================
    # IMAGE DECODING
    # =============================================================

    def _decode_image(
        self,
        image_bytes: bytes,
    ) -> np.ndarray:
        """
        Convert browser image bytes into an RGB NumPy image.

        Browser:
            canvas.toBlob()
                ↓
            JPEG bytes

        Backend:
            bytes
                ↓
            cv2.imdecode()
                ↓
            BGR
                ↓
            RGB
        """

        if not image_bytes:
            raise ValueError(
                "Image data is empty."
            )

        arr = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image_bgr = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR,
        )

        if image_bgr is None:
            raise ValueError(
                "Failed to decode image bytes; "
                "input is not a valid image."
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return image_rgb

    # =============================================================
    # PRIMARY FACE SELECTION
    # =============================================================

    def _select_primary_face(
        self,
        detections: list,
    ) -> Optional[dict]:
        """
        Select the largest detected face.

        This is useful when the webcam frame contains more than one
        face. The largest face is treated as the primary/user face.
        """

        if not detections:
            return None

        best_face = None
        best_area = -1

        for face in detections:
            try:
                box = face.get(
                    "box",
                    [],
                )

                if len(box) != 4:
                    continue

                _, _, width, height = box

                area = max(
                    0,
                    width,
                ) * max(
                    0,
                    height,
                )

                if area > best_area:
                    best_area = area
                    best_face = face

            except Exception:
                continue

        # If bounding boxes were malformed, fall back to the first
        # detection instead of failing the entire frame.
        if best_face is None:
            return detections[0]

        return best_face

    # =============================================================
    # LABEL NORMALIZATION
    # =============================================================

    def _normalize_label(
        self,
        label: str,
    ) -> str:
        """
        Convert labels from different models into the application's
        common six-emotion vocabulary.
        """

        if not label:
            return EmotionLabel.NEUTRAL.value

        normalized_label = (
            str(label)
            .strip()
            .lower()
        )

        label_map = {
            # Happy
            "joy": "happy",
            "happiness": "happy",
            "contentment": "happy",
            "content": "happy",
            "trust": "happy",

            # Sad
            "sadness": "sad",

            # Angry
            "anger": "angry",
            "disgust": "angry",

            # Fear
            "fear": "fear",

            # Surprise
            "surprise": "surprise",

            # Neutral
            "neutral": "neutral",
            "calm": "neutral",
            "anticipation": "neutral",
        }

        return label_map.get(
            normalized_label,
            normalized_label,
        )

    # =============================================================
    # MAP EMOTION SCORES
    # =============================================================

    def _map_label(
        self,
        scores: Dict[str, float],
    ) -> Tuple[str, float]:
        """
        Convert arbitrary emotion-score dictionaries into the
        application's six supported labels.

        Example FER output:

            {
                "angry": 0.05,
                "disgust": 0.01,
                "fear": 0.02,
                "happy": 0.80,
                "sad": 0.04,
                "surprise": 0.08
            }

        Result:

            ("happy", 0.80)
        """

        normalized: Dict[str, float] = {}

        for key, value in scores.items():
            try:
                label = self._normalize_label(
                    key
                )

                score = float(value)

                # Multiple labels can normalize into the same
                # application label. Keep the maximum score.
                normalized[label] = max(
                    normalized.get(label, 0.0),
                    score,
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        candidates = {
            EmotionLabel.HAPPY.value: normalized.get(
                EmotionLabel.HAPPY.value,
                0.0,
            ),
            EmotionLabel.SAD.value: normalized.get(
                EmotionLabel.SAD.value,
                0.0,
            ),
            EmotionLabel.ANGRY.value: normalized.get(
                EmotionLabel.ANGRY.value,
                0.0,
            ),
            EmotionLabel.FEAR.value: normalized.get(
                EmotionLabel.FEAR.value,
                0.0,
            ),
            EmotionLabel.SURPRISE.value: normalized.get(
                EmotionLabel.SURPRISE.value,
                0.0,
            ),
            EmotionLabel.NEUTRAL.value: normalized.get(
                EmotionLabel.NEUTRAL.value,
                0.0,
            ),
        }

        best_emotion, best_score = max(
            candidates.items(),
            key=lambda item: item[1],
        )

        return (
            best_emotion,
            float(best_score),
        )

    # =============================================================
    # UTILITY
    # =============================================================

    def get_supported_emotions(self) -> list:
        """
        Return the emotion vocabulary used by the application.
        """

        return [
            EmotionLabel.HAPPY.value,
            EmotionLabel.SAD.value,
            EmotionLabel.ANGRY.value,
            EmotionLabel.FEAR.value,
            EmotionLabel.SURPRISE.value,
            EmotionLabel.NEUTRAL.value,
        ]