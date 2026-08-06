import asyncio

from pipeline import build_pipeline
from processors.prompt_processor import PromptProcessor
from services.groq_service import GroqService


class DummyGroqService(GroqService):
    async def generate_response(self, prompt: str) -> str:
        return "Empathetic response."


def test_prompt_processor_builds_request() -> None:
    groq_service = DummyGroqService()
    processor = PromptProcessor(groq_service=groq_service)

    async def run_test() -> None:
        from pipecat import TextFrame, PipelineContext

        frame = TextFrame(payload="I feel anxious.")
        context = PipelineContext(data={"conversation_history": []})
        result = await processor.process(frame, context)
        assert result.payload == "Empathetic response."
        assert "generated_prompt" in result.metadata

    asyncio.run(run_test())
