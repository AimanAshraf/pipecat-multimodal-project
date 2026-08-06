import asyncio
from typing import List

from pipecat import Pipeline, PipelineRunner
from processors.context_processor import ConversationContextProcessor
from processors.deepgram_processor import DeepgramProcessor
from processors.emotion_fusion_processor import EmotionFusionProcessor
from processors.face_emotion_processor import FaceEmotionProcessor
from processors.logging_processor import LoggingProcessor
from processors.metrics_processor import MetricsProcessor
from processors.prompt_processor import PromptProcessor
from processors.response_emotion_processor import ResponseEmotionProcessor
from processors.text_emotion_processor import TextEmotionProcessor
from services.emotion_service import EmotionService
from services.groq_service import GroqService


def build_pipeline(
    emotion_service: EmotionService,
    groq_service: GroqService,
) -> Pipeline:
    processors = [
        MetricsProcessor(),
        LoggingProcessor(),
        DeepgramProcessor(),
        FaceEmotionProcessor(),
        TextEmotionProcessor(emotion_service=emotion_service),
        EmotionFusionProcessor(),
        ConversationContextProcessor(),
        PromptProcessor(groq_service=groq_service),
        ResponseEmotionProcessor(),
    ]
    return Pipeline(processors=processors)


async def create_runner(emotion_service: EmotionService, groq_service: GroqService) -> PipelineRunner:
    pipeline = build_pipeline(emotion_service=emotion_service, groq_service=groq_service)
    runner = PipelineRunner(pipeline=pipeline, max_workers=4)
    await runner.start()
    return runner
