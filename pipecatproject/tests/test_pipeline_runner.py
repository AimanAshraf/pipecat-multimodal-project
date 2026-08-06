import asyncio

from pipecat import AudioFrame, Frame, FrameProcessor, Pipeline, PipelineRunner


class DummyProcessor(FrameProcessor):
    async def process(self, frame: Frame, context: object) -> Frame:
        frame.metadata["processed"] = True
        return frame


def test_pipeline_runner_processes_frame() -> None:
    pipeline = Pipeline(processors=[DummyProcessor()])
    runner = PipelineRunner(pipeline=pipeline, max_workers=1)

    async def run_test() -> None:
        await runner.start()
        frame = AudioFrame(payload=b"test")
        task = await runner.submit(frame)
        await runner.queue.join()
        assert task.result is not None
        assert task.result.metadata["processed"] is True
        await runner.stop()

    asyncio.run(run_test())
