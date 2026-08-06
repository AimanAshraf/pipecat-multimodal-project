from models.emotion import EmotionLabel
from processors.emotion_fusion_processor import EmotionFusionProcessor
from pipecat import Frame


def test_emotion_fusion_defaults_to_neutral() -> None:
    processor = EmotionFusionProcessor()
    frame = Frame(payload=None, metadata={})
    fused = processor._fuse_emotions({}, {}, {})
    assert fused["emotion"] == EmotionLabel.NEUTRAL.value
    assert fused["confidence"] == 0.0


def test_emotion_fusion_combines_emotions() -> None:
    processor = EmotionFusionProcessor()
    fused = processor._fuse_emotions(
        {"emotion": "happy", "confidence": 0.9},
        {"emotion": "happy", "confidence": 0.6},
        {"emotion": "neutral", "confidence": 0.5},
    )
    assert fused["emotion"] == EmotionLabel.HAPPY.value
    assert fused["confidence"] > 0.0
