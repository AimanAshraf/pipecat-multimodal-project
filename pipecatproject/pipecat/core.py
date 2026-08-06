import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Frame:
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def copy(self) -> "Frame":
        return Frame(payload=self.payload, metadata=dict(self.metadata), frame_id=self.frame_id)


@dataclass
class AudioFrame(Frame):
    payload: bytes


@dataclass
class ImageFrame(Frame):
    payload: bytes


@dataclass
class TextFrame(Frame):
    payload: str


@dataclass
class LLMFrame(Frame):
    payload: str


@dataclass
class PipelineContext:
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTask:
    frame: Frame
    created_at: float = field(default_factory=time.time)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    completed_at: Optional[float] = None
    result: Optional[Frame] = None
    error: Optional[Exception] = None


class FrameProcessor(ABC):
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def process(self, frame: Frame, context: PipelineContext) -> Optional[Frame]:
        raise NotImplementedError


class Pipeline:
    def __init__(self, processors: List[FrameProcessor]) -> None:
        self.processors = processors

    async def run(self, frame: Frame, context: Optional[PipelineContext] = None) -> Optional[Frame]:
        context = context or PipelineContext()
        current_frame: Optional[Frame] = frame
        for processor in self.processors:
            if current_frame is None:
                break
            current_frame = await processor.process(current_frame, context)
        return current_frame


class PipelineRunner:
    def __init__(self, pipeline: Pipeline, max_workers: int = 4) -> None:
        self.pipeline = pipeline
        self.tasks: Dict[str, PipelineTask] = {}
        self.queue: asyncio.Queue[PipelineTask] = asyncio.Queue()
        self.max_workers = max_workers
        self._workers: List[asyncio.Task[Any]] = []
        self.context = PipelineContext()
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for _ in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

    async def stop(self) -> None:
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()

    async def _worker_loop(self) -> None:
        while self._running:
            task = await self.queue.get()
            try:
                task.result = await self.pipeline.run(task.frame, self.context)
                task.completed_at = time.time()
            except Exception as exc:
                task.error = exc
            finally:
                self.queue.task_done()

    async def submit(self, frame: Frame) -> PipelineTask:
        task = PipelineTask(frame=frame)
        self.tasks[task.task_id] = task
        await self.queue.put(task)
        return task

    async def run(self, frame: Frame) -> Optional[Frame]:
        return await self.pipeline.run(frame, self.context)
