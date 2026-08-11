from typing import Optional

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

from utils.logger import get_logger


logger = get_logger(__name__)


def build_pipeline(
    emotion_service: EmotionService,
    groq_service: GroqService,
) -> Pipeline:
    """
    Build the normal request-processing pipeline.

    This pipeline is used for:
        - /api/text
        - /api/audio
        - /api/image
        - /api/multimodal

    The live WebSocket 2-second emotion loop is handled separately
    by app.py.
    """

    processors = [
        # ---------------------------------------------------------
        # Monitoring
        # ---------------------------------------------------------

        MetricsProcessor(),

        LoggingProcessor(),

        # ---------------------------------------------------------
        # Input processing
        # ---------------------------------------------------------

        DeepgramProcessor(),

        FaceEmotionProcessor(
            emotion_service=emotion_service,
        ),

        TextEmotionProcessor(
            emotion_service=emotion_service,
        ),

        # ---------------------------------------------------------
        # Emotion fusion
        # ---------------------------------------------------------

        EmotionFusionProcessor(),

        # ---------------------------------------------------------
        # Conversation / LLM
        # ---------------------------------------------------------

        ConversationContextProcessor(),

        PromptProcessor(
            groq_service=groq_service,
        ),

        # ---------------------------------------------------------
        # Response emotion / TTS metadata
        # ---------------------------------------------------------

        ResponseEmotionProcessor(),
    ]

    logger.info(
        "Building emotion-aware pipeline with %d processors",
        len(processors),
    )

    return Pipeline(
        processors=processors,
    )


async def create_runner(
    emotion_service: EmotionService,
    groq_service: GroqService,
) -> PipelineRunner:
    """
    Create and start the shared PipelineRunner.

    max_workers=2 keeps concurrent processing bounded, which is
    particularly important because the emotion models can consume
    significant memory.
    """

    pipeline = build_pipeline(
        emotion_service=emotion_service,
        groq_service=groq_service,
    )

    runner = PipelineRunner(
        pipeline=pipeline,
        max_workers=2,
    )

    await runner.start()

    logger.info(
        "Pipeline runner started"
    )

    return runner