import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pipecat import AudioFrame, ImageFrame, TextFrame
from pipeline import create_runner
from processors.emotion_fusion_processor import EmotionFusionProcessor
from services.deepgram_service import DeepgramService
from services.emotion_service import EmotionService
from services.groq_service import GroqService
from utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Emotion-Aware Conversational AI Assistant")

frontend_dir = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

emotion_service: Optional[EmotionService] = None
runner = None


# ── Shared streaming state ─────────────────────────────────────────────────
# WebSocket endpoints write here; /api/multimodal reads from here so it always
# has the latest browser-streamed face and speech data available.
@dataclass
class StreamingState:
    face_emotion: Dict[str, Any] = field(default_factory=lambda: {"emotion": "neutral", "confidence": 0.0})
    speech_emotion: Dict[str, Any] = field(default_factory=lambda: {"emotion": "neutral", "confidence": 0.0})
    transcript: str = ""


_stream_state = StreamingState()


# ── Helpers ────────────────────────────────────────────────────────────────
def _assert_task_success(task):
    if task.error:
        logger.error("Pipeline task %s failed", task.task_id, exc_info=task.error)
        raise HTTPException(status_code=500, detail=str(task.error))
    if task.result is None:
        logger.error("Pipeline task %s completed without a result", task.task_id)
        raise HTTPException(status_code=500, detail="Pipeline task completed without a result")
    return task.result


# ── Lifecycle ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event() -> None:
    global runner, emotion_service
    emotion_service = EmotionService(model_name="j-hartmann/emotion-english-distilroberta-base")
    groq_service = GroqService()
    runner = await create_runner(emotion_service=emotion_service, groq_service=groq_service)
    logger.info("Pipeline runner started")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global runner
    if runner:
        await runner.stop()
        logger.info("Pipeline runner stopped")


# ── Static / UI ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index() -> Any:
    file_path = frontend_dir / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(file_path)


# ── REST endpoints (unchanged) ─────────────────────────────────────────────
@app.post("/api/audio")
async def process_audio(request: Request) -> JSONResponse:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    frame = AudioFrame(payload=body)
    task = await runner.submit(frame)
    await runner.queue.join()
    result_frame = runner.tasks[task.task_id].result

    response_payload = ""
    transcript = ""
    if isinstance(result_frame, TextFrame):
        response_payload = result_frame.payload or ""
        transcript = result_frame.metadata.get("transcript", "")

    return JSONResponse({
        "task_id": task.task_id,
        "response": response_payload,
        "transcript": transcript,
        "metadata": result_frame.metadata if result_frame else {},
    })


@app.post("/api/image")
async def process_image(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
            upload = form.get("image")
            if upload is None:
                raise HTTPException(status_code=400, detail="Missing image file in form data")
            body = await upload.read()
        except AssertionError:
            raise HTTPException(
                status_code=400,
                detail="python-multipart is not installed. Send raw image bytes instead.",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse form data: {exc}")

    if not body:
        raise HTTPException(status_code=400, detail="Empty image payload")

    frame = ImageFrame(payload=body)
    task = await runner.submit(frame)
    await runner.queue.join()
    result_frame = runner.tasks[task.task_id].result
    return JSONResponse({"task_id": task.task_id, "metadata": result_frame.metadata if result_frame else {}})


@app.post("/api/text")
async def process_text(payload: Dict[str, str]) -> JSONResponse:
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text payload")
    frame = TextFrame(payload=text)
    task = await runner.submit(frame)
    await runner.queue.join()
    task_result = runner.tasks[task.task_id]
    result_frame = _assert_task_success(task_result)

    response_text = getattr(result_frame, "payload", "")
    if isinstance(response_text, str):
        response_text = response_text.strip()
    if not response_text:
        response_text = (
            result_frame.metadata.get("response_text")
            or result_frame.metadata.get("generated_text")
            or result_frame.metadata.get("text")
            or "Sorry, I couldn't generate a response at the moment."
        )
        if isinstance(response_text, str):
            response_text = response_text.strip()
        if not response_text:
            response_text = "Sorry, I couldn't generate a response at the moment."
        logger.warning("/api/text returning fallback response because task payload was empty")

    return JSONResponse({
        "task_id": task.task_id,
        "response": response_text,
        "metadata": result_frame.metadata if result_frame else {},
    })


# ── WebSocket: continuous video frame streaming ────────────────────────────
@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket) -> None:
    """
    Receives JPEG frames from the browser (~300-500 ms cadence).
    Runs face emotion detection and stores the result in shared state so
    /api/multimodal always has the latest face reading available.
    Returns the prediction to the frontend after each frame.
    No cv2.VideoCapture — all frames come from the browser.
    """
    await websocket.accept()
    logger.info("WebSocket /ws/video connected")
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            if not frame_bytes:
                continue

            emotion_label, confidence = emotion_service.analyze_face_emotion(frame_bytes)

            # Update shared state for multimodal fusion
            _stream_state.face_emotion = {
                "emotion": emotion_label,
                "confidence": round(float(confidence), 4),
            }

            await websocket.send_json({"face_emotion": _stream_state.face_emotion})

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/video disconnected")
    except Exception as exc:
        logger.error("WebSocket /ws/video error: %s", exc, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ── WebSocket: continuous audio chunk streaming ────────────────────────────
@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    """
    Receives small MediaRecorder chunks (~500-1000 ms) from the browser.
    Uses DeepgramService to transcribe each chunk, then runs speech emotion
    classification via the existing EmotionService. Stores the latest
    transcript and speech emotion in shared state for multimodal fusion.
    Does NOT run the full Pipecat pipeline or generate AI responses here —
    that happens in /api/multimodal when the user explicitly triggers it.
    Returns the transcript + speech emotion back to the frontend.
    No PyAudio or server-side mic access.
    """
    await websocket.accept()
    logger.info("WebSocket /ws/audio connected")
    deepgram_service = DeepgramService()
    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            if not audio_bytes:
                continue

            # Transcribe via Deepgram
            try:
                transcript_payload = await deepgram_service.transcribe(audio_bytes)
                transcript = deepgram_service.parse_transcript(transcript_payload)
            except Exception as exc:
                logger.warning("Deepgram transcription failed in WS: %s", exc)
                transcript = ""

            # Speech emotion classification on transcript text
            speech_label, speech_confidence = ("neutral", 0.0)
            if transcript.strip():
                speech_label, speech_confidence = emotion_service.classify_text_emotion(transcript)

            # Update shared state for multimodal fusion
            _stream_state.transcript = transcript
            _stream_state.speech_emotion = {
                "emotion": speech_label,
                "confidence": round(float(speech_confidence), 4),
            }

            await websocket.send_json({
                "transcript": transcript,
                "speech_emotion": _stream_state.speech_emotion,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket /ws/audio disconnected")
    except Exception as exc:
        logger.error("WebSocket /ws/audio error: %s", exc, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ── Multimodal endpoint ────────────────────────────────────────────────────
@app.post("/api/multimodal")
async def process_multimodal(request: Request) -> JSONResponse:
    """
    Weighted fusion: face=0.55, text=0.30, speech=0.15.

    Accepts explicit image/audio/text via multipart form (original behaviour).
    If the browser has an active WebSocket stream, the latest face and speech
    emotion from _stream_state are used as fallback / supplement so the
    multimodal endpoint always has up-to-date predictions even when the user
    did not upload a fresh frame or audio blob for this specific request.
    """
    global emotion_service

    if emotion_service is None:
        raise HTTPException(status_code=500, detail="Emotion service is unavailable")

    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail="/api/multimodal requires multipart/form-data")

    try:
        form = await request.form()
        text = (form.get("text") or "").strip()
        image_upload = form.get("image")
        audio_upload = form.get("audio")
        image_bytes = await image_upload.read() if image_upload is not None else b""
        audio_bytes = await audio_upload.read() if audio_upload is not None else b""
    except AssertionError:
        raise HTTPException(
            status_code=400,
            detail="python-multipart is not installed. Send multipart/form-data with image/audio/text fields.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse multipart form data: {exc}")

    if not text:
        raise HTTPException(status_code=400, detail="text field is required.")

    # ── Face emotion: use uploaded image or fall back to latest WS state ──
    if image_bytes:
        face_label, face_conf = emotion_service.analyze_face_emotion(image_bytes)
        face_emotion = {"emotion": face_label, "confidence": float(face_conf)}
    else:
        face_emotion = _stream_state.face_emotion

    # ── Speech emotion: use uploaded audio or fall back to latest WS state ─
    if audio_bytes:
        deepgram_service = DeepgramService()
        transcript_payload = await deepgram_service.transcribe(audio_bytes)
        transcript = deepgram_service.parse_transcript(transcript_payload)
        speech_label, speech_conf = emotion_service.classify_text_emotion(transcript) if transcript.strip() else ("neutral", 0.0)
        speech_sentiment = {"emotion": speech_label, "confidence": float(speech_conf)}
    else:
        transcript = _stream_state.transcript
        speech_sentiment = _stream_state.speech_emotion

    # ── Text emotion (always from the text field) ──────────────────────────
    text_label, text_conf = emotion_service.classify_text_emotion(text)
    text_emotion = {"emotion": text_label, "confidence": float(text_conf)}

    # ── Weighted fusion (face=0.55, text=0.30, speech=0.15) ───────────────
    fusion = EmotionFusionProcessor()._fuse_emotions(face_emotion, text_emotion, speech_sentiment)

    multimodal_frame = TextFrame(payload=text, metadata={
        "source": "multimodal",
        "face_emotion": face_emotion,
        "speech_sentiment": speech_sentiment,
        "text_emotion": text_emotion,
        "transcript": transcript,
        "fused_emotion": fusion,
    })

    task = await runner.submit(multimodal_frame)
    await runner.queue.join()
    task_result = runner.tasks[task.task_id]
    result_frame = _assert_task_success(task_result)

    response_text = getattr(result_frame, "payload", "")
    if isinstance(response_text, str):
        response_text = response_text.strip()
    if not response_text:
        response_text = (
            result_frame.metadata.get("response_text")
            or result_frame.metadata.get("generated_text")
            or result_frame.metadata.get("text")
            or "Sorry, I couldn't generate a response at the moment."
        )
        if isinstance(response_text, str):
            response_text = response_text.strip()
        if not response_text:
            response_text = "Sorry, I couldn't generate a response at the moment."

    return JSONResponse({
        "task_id": task.task_id,
        "response": response_text,
        "transcript": transcript,
        "metadata": result_frame.metadata if result_frame else {},
    })
