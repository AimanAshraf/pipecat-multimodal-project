import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pipecat import AudioFrame, ImageFrame, TextFrame
from pipeline import create_runner
from processors.emotion_fusion_processor import EmotionFusionProcessor
from services.cartesia_service import CartesiaService
from services.deepgram_service import DeepgramService
from services.emotion_service import EmotionService
from services.groq_service import GroqService
from utils.logger import get_logger

emotion_service: Optional[EmotionService] = None

logger = get_logger(__name__)
app = FastAPI(title="Emotion-Aware Conversational AI Assistant")

frontend_dir = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

runner = None


def _assert_task_success(task):
    if task.error:
        logger.error("Pipeline task %s failed", task.task_id, exc_info=task.error)
        raise HTTPException(status_code=500, detail=str(task.error))
    if task.result is None:
        logger.error("Pipeline task %s completed without a result", task.task_id)
        raise HTTPException(status_code=500, detail="Pipeline task completed without a result")
    return task.result


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


@app.get("/", response_class=HTMLResponse)
async def index() -> Any:
    file_path = frontend_dir / "index.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(file_path)


@app.post("/api/audio")
async def process_audio(request: Request) -> JSONResponse:
    global runner
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
    global runner
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
                detail="python-multipart is not installed. Send raw image bytes instead of multipart form data.",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse multipart form data: {exc}")

    if not body:
        raise HTTPException(status_code=400, detail="Empty image payload")

    frame = ImageFrame(payload=body)
    task = await runner.submit(frame)
    await runner.queue.join()
    result_frame = runner.tasks[task.task_id].result
    return JSONResponse({"task_id": task.task_id, "metadata": result_frame.metadata if result_frame else {}})


@app.post("/api/text")
async def process_text(payload: Dict[str, str]) -> JSONResponse:
    global runner
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text payload")
    frame = TextFrame(payload=text)
    task = await runner.submit(frame)
    await runner.queue.join()
    task_result = runner.tasks[task.task_id]
    result_frame = _assert_task_success(task_result)

    payload = getattr(result_frame, "payload", "")
    if isinstance(payload, str):
        payload = payload.strip()
    # Prefer explicit payload, but fall back to metadata-stored response_text for robustness
    if payload:
        response_text = payload
    else:
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
    if not payload:
        logger.warning(
            "/api/text returning fallback response because task payload was empty or whitespace",
        )

    return JSONResponse({
        "task_id": task.task_id,
        "response": response_text,
        "metadata": result_frame.metadata if result_frame else {},
    })


@app.post("/api/multimodal")
async def process_multimodal(request: Request) -> JSONResponse:
    global runner, emotion_service
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

    if not image_bytes or not audio_bytes or not text:
        raise HTTPException(status_code=400, detail="Image, audio, and text are all required for multimodal input.")

    face_emotion_label, face_confidence = emotion_service.analyze_face_emotion(image_bytes)
    face_emotion = {
        "emotion": face_emotion_label,
        "confidence": float(face_confidence),
    }

    deepgram_service = DeepgramService()
    transcript_payload = await deepgram_service.transcribe(audio_bytes)
    transcript = deepgram_service.parse_transcript(transcript_payload)
    speech_label, speech_confidence = emotion_service.classify_text_emotion(transcript)
    speech_sentiment = {
        "emotion": speech_label,
        "confidence": float(speech_confidence),
    }

    text_label, text_confidence = emotion_service.classify_text_emotion(text)
    text_emotion = {
        "emotion": text_label,
        "confidence": float(text_confidence),
    }

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

    payload = getattr(result_frame, "payload", "")
    if isinstance(payload, str):
        payload = payload.strip()
    if payload:
        response_text = payload
    else:
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
