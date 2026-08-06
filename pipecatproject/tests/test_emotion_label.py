from models.emotion import EmotionLabel


def test_emotion_label_values() -> None:
    assert EmotionLabel.HAPPY.value == "happy"
    assert EmotionLabel.SAD.value == "sad"
    assert EmotionLabel.NEUTRAL.value == "neutral"
