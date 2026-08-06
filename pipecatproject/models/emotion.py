from enum import Enum


class EmotionLabel(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"
