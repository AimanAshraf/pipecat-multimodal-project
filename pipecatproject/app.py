import asyncio
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
)

from fastapi.staticfiles import StaticFiles

from pipecat import (
    AudioFrame,
    ImageFrame,
    TextFrame,
)

from pipeline import create_runner

from services.deepgram_service import DeepgramService
from services.emotion_service import EmotionService
from services.groq_service import GroqService

from utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="Emotion-Aware Conversational AI Assistant"
)


# ============================================================================
# FRONTEND
# ============================================================================

frontend_dir = (
    Path(__file__).resolve().parent / "frontend"
)

app.mount(
    "/static",
    StaticFiles(directory=frontend_dir),
    name="static",
)


# ============================================================================
# GLOBAL SERVICES
# ============================================================================

emotion_service: Optional[EmotionService] = None

groq_service: Optional[GroqService] = None

runner = None


# ============================================================================
# LIVE EMOTION CONFIGURATION
# ============================================================================

# Analyze the current live stream every 4 seconds.

EMOTION_WINDOW_SECONDS = 4.0


# Original project weights:
#
# face   = 0.55
# text   = 0.30
# speech = 0.15
#
# Live mode has no manually entered text, so we normalize the
# available face + speech weights.

LIVE_FACE_WEIGHT = 0.55
LIVE_SPEECH_WEIGHT = 0.15

LIVE_TOTAL_WEIGHT = (
    LIVE_FACE_WEIGHT +
    LIVE_SPEECH_WEIGHT
)

LIVE_FACE_NORMALIZED_WEIGHT = (
    LIVE_FACE_WEIGHT /
    LIVE_TOTAL_WEIGHT
)

LIVE_SPEECH_NORMALIZED_WEIGHT = (
    LIVE_SPEECH_WEIGHT /
    LIVE_TOTAL_WEIGHT
)


# ============================================================================
# SHARED STREAMING STATE
# ============================================================================

@dataclass
class StreamingState:
    """
    State shared by the live video/audio WebSockets and the
    background 2-second emotion analyzer.
    """

    # Latest facial emotion.
    face_emotion: Dict[str, Any] = field(
        default_factory=lambda: {
            "emotion": "neutral",
            "confidence": 0.0,
        }
    )

    # Latest speech emotion.
    speech_emotion: Dict[str, Any] = field(
        default_factory=lambda: {
            "emotion": "neutral",
            "confidence": 0.0,
        }
    )

    # Most recent transcript chunk.
    transcript: str = ""

    # All transcript chunks received during the
    # current 2-second window.
    transcript_chunks: list[str] = field(
        default_factory=list
    )

    # Most recent fused live emotion.
    fused_emotion: Dict[str, Any] = field(
        default_factory=lambda: {
            "emotion": "neutral",
            "confidence": 0.0,
            "scores": {
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "neutral": 0.0,
            },
        }
    )

    # Incremented after every 2-second window.
    window_id: int = 0


_stream_state = StreamingState()


# ============================================================================
# ACTIVE WEBSOCKET CLIENTS
# ============================================================================

_video_clients: Set[WebSocket] = set()

_audio_clients: Set[WebSocket] = set()


# ============================================================================
# STATE LOCK
# ============================================================================

_stream_lock = asyncio.Lock()


# ============================================================================
# BACKGROUND TASK
# ============================================================================

_emotion_window_task: Optional[asyncio.Task] = None


# ============================================================================
# EMOTION HELPERS
# ============================================================================

def _normalize_confidence(
    value: Any,
) -> float:
    """
    Convert a confidence value to a float in [0, 1].
    """

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def _normalize_emotion(
    emotion: Any,
) -> str:
    """
    Normalize emotion labels produced by different models.
    """

    if emotion is None:
        return "neutral"

    value = str(
        emotion
    ).strip().lower()

    aliases = {
        "joy": "happy",
        "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
        "calm": "neutral",
        "contentment": "happy",
        "trust": "happy",
        "disgust": "angry",
        "anticipation": "neutral",
    }

    return aliases.get(
        value,
        value,
    )


def _empty_emotion_scores() -> Dict[str, float]:
    """
    Create a fresh emotion score dictionary.
    """

    return {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "neutral": 0.0,
    }


def _fuse_live_face_and_speech(
    face_emotion: Dict[str, Any],
    speech_emotion: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fuse facial and speech emotion for the automatic live
    2-second window.

    Original weights:

        face   = 0.55
        text   = 0.30
        speech = 0.15

    Since live mode has no manually entered text:

        face   = 0.55 / 0.70
        speech = 0.15 / 0.70

    This prevents the missing text modality from reducing the
    live confidence simply because text isn't present.
    """

    face_label = _normalize_emotion(
        face_emotion.get(
            "emotion",
            "neutral",
        )
    )

    face_confidence = _normalize_confidence(
        face_emotion.get(
            "confidence",
            0.0,
        )
    )

    speech_label = _normalize_emotion(
        speech_emotion.get(
            "emotion",
            "neutral",
        )
    )

    speech_confidence = _normalize_confidence(
        speech_emotion.get(
            "confidence",
            0.0,
        )
    )

    scores = _empty_emotion_scores()

    # ------------------------------------------------------------
    # FACE
    # ------------------------------------------------------------

    if face_label in scores:
        scores[face_label] += (
            face_confidence *
            LIVE_FACE_NORMALIZED_WEIGHT
        )

    # ------------------------------------------------------------
    # SPEECH
    # ------------------------------------------------------------

    if speech_label in scores:
        scores[speech_label] += (
            speech_confidence *
            LIVE_SPEECH_NORMALIZED_WEIGHT
        )

    # ------------------------------------------------------------
    # DOMINANT EMOTION
    # ------------------------------------------------------------

    # CRITICAL FIX:
    # When there is no evidence from any modality, all scores
    # are 0.0. In this case, do NOT return "happy" merely
    # because it is the first dictionary key.

    max_score = max(scores.values()) if scores else 0.0

    if max_score <= 0.0:
        # No evidence from any modality
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "scores": {
                key: 0.0
                for key in scores.keys()
            },
            "has_evidence": False,
        }

    dominant_emotion = max(
        scores,
        key=scores.get,
    )

    confidence = _normalize_confidence(
        scores[dominant_emotion]
    )

    return {
        "emotion": dominant_emotion,
        "confidence": round(
            confidence,
            4,
        ),
        "scores": {
            key: round(
                value,
                4,
            )
            for key, value in scores.items()
        },
        "has_evidence": True,
    }


# ============================================================================
# SAFE WEBSOCKET SEND
# ============================================================================

async def _safe_send_json(
    websocket: WebSocket,
    payload: Dict[str, Any],
) -> bool:
    """
    Send JSON without allowing a disconnected browser to crash
    the background emotion loop.
    """

    try:
        await websocket.send_json(
            payload
        )

        return True

    except Exception as exc:
        logger.debug(
            "Could not send WebSocket message: %s",
            exc,
        )

        return False


async def _broadcast_live_emotion(
    payload: Dict[str, Any],
) -> None:
    """
    Send the 4-second emotion result to connected video/audio
    clients.
    """

    clients = list(
        _video_clients
    ) + list(
        _audio_clients
    )

    if not clients:
        logger.debug(
            "No clients connected to broadcast emotion result"
        )
        return

    logger.info(
        "Broadcasting emotion result to %d client(s)",
        len(clients)
    )

    disconnected = []

    for websocket in clients:

        success = await _safe_send_json(
            websocket,
            payload,
        )

        if not success:
            disconnected.append(
                websocket
            )

    for websocket in disconnected:
        _video_clients.discard(
            websocket
        )

        _audio_clients.discard(
            websocket
        )


# ============================================================================
# PIPELINE RESULT VALIDATION
# ============================================================================

def _assert_task_success(task):
    """
    Verify that a pipeline task completed successfully.
    """

    if task.error:

        logger.error(
            "Pipeline task %s failed",
            task.task_id,
            exc_info=task.error,
        )

        raise HTTPException(
            status_code=500,
            detail=str(
                task.error
            ),
        )

    if task.result is None:

        logger.error(
            "Pipeline task %s completed without a result",
            task.task_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Pipeline task completed "
                "without a result"
            ),
        )

    return task.result


# ============================================================================
# 2-SECOND LIVE EMOTION PROCESSOR
# ============================================================================

async def _generate_empathetic_response(
    fused_emotion: Dict[str, Any],
    transcript: str,
    face_emotion: Dict[str, Any],
    speech_emotion: Dict[str, Any],
    groq_service,
) -> str:
    """
    Generate an AI response based on the detected emotions and transcript
    collected over a 4-second window.
    """

    emotion = fused_emotion.get("emotion", "neutral")
    confidence = fused_emotion.get("confidence", 0.0)
    has_evidence = fused_emotion.get("has_evidence", False)

    # If no evidence, return a waiting message
    if not has_evidence or confidence < 0.1:
        return "I'm here and listening. Feel free to share what's on your mind."

    # Build context for the AI
    emotion_context = f"The user appears to be feeling {emotion}"
    
    if transcript:
        emotion_context += f" and said: '{transcript}'"
    
    face_conf = face_emotion.get("confidence", 0.0)
    speech_conf = speech_emotion.get("confidence", 0.0)
    
    if face_conf > 0.3:
        emotion_context += f". Their facial expression shows {face_emotion.get('emotion', 'neutral')}"
    
    if speech_conf > 0.3 and transcript:
        emotion_context += f", and their speech tone suggests {speech_emotion.get('emotion', 'neutral')}"

    # Create prompt for Groq
    prompt = f"""You are an empathetic AI assistant analyzing real-time emotions.

Context: {emotion_context}

Generate a brief (1-2 sentences), supportive response that:
1. Acknowledges their emotional state
2. Shows empathy
3. Is conversational and natural
4. Encourages continued interaction if appropriate

Response:"""

    try:
        # Use Groq service to generate response
        response = await groq_service.generate(prompt)
        
        if response and isinstance(response, str):
            return response.strip()
        
        # Fallback responses based on emotion
        fallback_responses = {
            "happy": "That's wonderful! Your positive energy is contagious. What's bringing you joy?",
            "sad": "I sense you might be feeling down. I'm here to listen if you'd like to talk about it.",
            "angry": "I notice some frustration. Take a deep breath - I'm here to help however I can.",
            "fear": "It seems like something is concerning you. Remember, it's okay to feel this way, and you're not alone.",
            "surprise": "Something unexpected happened? I'm curious to hear more about what caught your attention.",
            "neutral": "I'm here with you. Feel free to share whatever's on your mind."
        }
        
        return fallback_responses.get(emotion, "I'm here and listening to you.")
        
    except Exception as exc:
        logger.error(
            "AI response generation failed: %s",
            exc,
            exc_info=True,
        )
        
        # Simple fallback
        return f"I notice you're feeling {emotion}. I'm here to support you."


async def _process_emotion_window() -> None:
    """
    Process one live 4-second window.

    Steps:

        1. Snapshot current face emotion.
        2. Snapshot transcript chunks.
        3. Combine transcript chunks.
        4. Calculate speech emotion from the combined transcript.
        5. Fuse face + speech.
        6. Generate AI response based on emotions.
        7. Store result.
        8. Broadcast result to frontend.
        9. Clear transcript chunks for the next window.
    """

    global _stream_state

    # Skip processing if no clients are connected
    if not _video_clients and not _audio_clients:
        logger.debug(
            "Skipping emotion window - no clients connected"
        )
        return

    # ------------------------------------------------------------
    # SNAPSHOT CURRENT WINDOW
    # ------------------------------------------------------------

    async with _stream_lock:

        face_emotion = dict(
            _stream_state.face_emotion
        )

        transcript_chunks = list(
            _stream_state.transcript_chunks
        )

        window_id = (
            _stream_state.window_id + 1
        )

        # Clear the current transcript window immediately.
        #
        # New audio arriving after this point belongs to the
        # next 4-second window.

        _stream_state.transcript_chunks.clear()

        _stream_state.window_id = (
            window_id
        )

    # ------------------------------------------------------------
    # COMBINE TRANSCRIPT
    # ------------------------------------------------------------

    window_transcript = " ".join(
        chunk.strip()
        for chunk in transcript_chunks
        if chunk and chunk.strip()
    ).strip()

    # ------------------------------------------------------------
    # SPEECH EMOTION
    # ------------------------------------------------------------

    speech_emotion = {
        "emotion": "neutral",
        "confidence": 0.0,
    }

    if window_transcript:

        try:

            speech_label, speech_confidence = (
                emotion_service.classify_text_emotion(
                    window_transcript
                )
            )

            speech_emotion = {
                "emotion": speech_label,
                "confidence": round(
                    float(
                        speech_confidence
                    ),
                    4,
                ),
            }

        except Exception as exc:

            logger.error(
                "Live speech emotion classification failed: %s",
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------
    # STORE LATEST SPEECH EMOTION
    # ------------------------------------------------------------

    async with _stream_lock:

        _stream_state.speech_emotion = (
            speech_emotion
        )

        if window_transcript:
            _stream_state.transcript = (
                window_transcript
            )

    # ------------------------------------------------------------
    # FACE + SPEECH FUSION
    # ------------------------------------------------------------

    fused_emotion = (
        _fuse_live_face_and_speech(
            face_emotion=face_emotion,
            speech_emotion=speech_emotion,
        )
    )

    # ------------------------------------------------------------
    # GENERATE AI RESPONSE
    # ------------------------------------------------------------

    logger.info(
        "Generating AI response for emotion=%s, confidence=%.2f, transcript='%s'",
        fused_emotion.get("emotion"),
        fused_emotion.get("confidence", 0.0),
        window_transcript[:50] if window_transcript else ""
    )

    ai_response = await _generate_empathetic_response(
        fused_emotion=fused_emotion,
        transcript=window_transcript,
        face_emotion=face_emotion,
        speech_emotion=speech_emotion,
        groq_service=groq_service,
    )

    logger.info(
        "AI response generated: %s",
        ai_response[:100] + "..." if len(ai_response) > 100 else ai_response
    )

    # ------------------------------------------------------------
    # STORE FUSED EMOTION
    # ------------------------------------------------------------

    async with _stream_lock:

        _stream_state.fused_emotion = (
            fused_emotion
        )

    # ------------------------------------------------------------
    # RESULT PAYLOAD
    # ------------------------------------------------------------

    payload = {
        "type": "live_emotion",

        "window_id": window_id,

        "window_duration": (
            EMOTION_WINDOW_SECONDS
        ),

        "ai_response": ai_response,

        "transcript": window_transcript,

        "timestamp": time.time(),
    }

    # Only include face_emotion if video is streaming
    if len(_video_clients) > 0:
        payload["face_emotion"] = face_emotion

    # Only include speech_emotion if audio is streaming
    if len(_audio_clients) > 0:
        payload["speech_emotion"] = speech_emotion

    # Always include fused_emotion
    payload["fused_emotion"] = fused_emotion

    # ------------------------------------------------------------
    # SEND TO FRONTEND
    # ------------------------------------------------------------

    logger.info(
        "About to broadcast to clients. Connected clients: video=%d, audio=%d",
        len(_video_clients),
        len(_audio_clients)
    )

    await _broadcast_live_emotion(
        payload
    )

    logger.info(
        (
            "[FUSION] LIVE EMOTION | "
            "window=%s | "
            "face=%s | "
            "speech=%s | "
            "fused=%s | "
            "transcript=%s | "
            "ai_response=%s"
        ),
        window_id,
        face_emotion,
        speech_emotion,
        fused_emotion,
        window_transcript,
        ai_response[:50] + "..." if len(ai_response) > 50 else ai_response,
    )


# ============================================================================
# BACKGROUND 2-SECOND LOOP
# ============================================================================

async def _emotion_window_loop() -> None:
    """
    Continuously process the live stream every 4 seconds.
    """

    logger.info(
        "Live 4-second emotion window loop started"
    )

    try:

        while True:

            await asyncio.sleep(
                EMOTION_WINDOW_SECONDS
            )

            try:

                await _process_emotion_window()

            except asyncio.CancelledError:

                raise

            except Exception as exc:

                logger.error(
                    "Live emotion window failed: %s",
                    exc,
                    exc_info=True,
                )

    except asyncio.CancelledError:

        logger.info(
            "Live 4-second emotion window loop stopped"
        )

        raise


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event() -> None:

    global runner
    global emotion_service
    global groq_service
    global _emotion_window_task

    logger.info(
        "Starting Emotion-Aware Conversational AI Assistant"
    )

    # ------------------------------------------------------------
    # EMOTION SERVICE
    # ------------------------------------------------------------

    emotion_service = EmotionService(
        model_name=(
            "j-hartmann/"
            "emotion-english-distilroberta-base"
        )
    )

    # ------------------------------------------------------------
    # GROQ
    # ------------------------------------------------------------

    groq_service = GroqService()

    # ------------------------------------------------------------
    # PIPECAT RUNNER
    # ------------------------------------------------------------

    runner = await create_runner(
        emotion_service=emotion_service,
        groq_service=groq_service,
    )

    logger.info(
        "Pipeline runner started"
    )

    # ------------------------------------------------------------
    # LIVE 2-SECOND ANALYZER
    # ------------------------------------------------------------

    _emotion_window_task = (
        asyncio.create_task(
            _emotion_window_loop()
        )
    )

    logger.info(
        "Live 4-second emotion analyzer started"
    )


# ============================================================================
# APPLICATION SHUTDOWN
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event() -> None:

    global runner
    global _emotion_window_task

    # ------------------------------------------------------------
    # STOP LIVE ANALYZER
    # ------------------------------------------------------------

    if _emotion_window_task:

        _emotion_window_task.cancel()

        try:

            await _emotion_window_task

        except asyncio.CancelledError:

            pass

        _emotion_window_task = None

    # ------------------------------------------------------------
    # STOP PIPECAT RUNNER
    # ------------------------------------------------------------

    if runner:

        await runner.stop()

        logger.info(
            "Pipeline runner stopped"
        )


# ============================================================================
# FRONTEND
# ============================================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def index() -> Any:

    file_path = (
        frontend_dir /
        "index.html"
    )

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend not found",
        )

    return FileResponse(
        file_path
    )


# ============================================================================
# REST: AUDIO
# ============================================================================

@app.post("/api/audio")
async def process_audio(
    request: Request,
) -> JSONResponse:

    body = await request.body()

    if not body:

        raise HTTPException(
            status_code=400,
            detail="Empty audio payload",
        )

    frame = AudioFrame(
        payload=body
    )

    task = await runner.submit(
        frame
    )

    await runner.queue.join()

    result_frame = (
        runner.tasks[
            task.task_id
        ].result
    )

    response_payload = ""
    transcript = ""

    if isinstance(
        result_frame,
        TextFrame,
    ):

        response_payload = (
            result_frame.payload
            or ""
        )

        transcript = (
            result_frame.metadata.get(
                "transcript",
                "",
            )
        )

    return JSONResponse(
        {
            "task_id": task.task_id,
            "response": response_payload,
            "transcript": transcript,
            "metadata": (
                result_frame.metadata
                if result_frame
                else {}
            ),
        }
    )


# ============================================================================
# REST: IMAGE
# ============================================================================

@app.post("/api/image")
async def process_image(
    request: Request,
) -> JSONResponse:

    content_type = request.headers.get(
        "content-type",
        "",
    )

    body = await request.body()

    # ------------------------------------------------------------
    # MULTIPART IMAGE
    # ------------------------------------------------------------

    if content_type.startswith(
        "multipart/form-data"
    ):

        try:

            form = await request.form()

            upload = form.get(
                "image"
            )

            if upload is None:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Missing image file "
                        "in form data"
                    ),
                )

            body = await upload.read()

        except AssertionError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "python-multipart is not installed. "
                    "Send raw image bytes instead."
                ),
            )

        except HTTPException:

            raise

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not parse form data: "
                    f"{exc}"
                ),
            )

    if not body:

        raise HTTPException(
            status_code=400,
            detail="Empty image payload",
        )

    frame = ImageFrame(
        payload=body
    )

    task = await runner.submit(
        frame
    )

    await runner.queue.join()

    result_frame = (
        runner.tasks[
            task.task_id
        ].result
    )

    return JSONResponse(
        {
            "task_id": task.task_id,
            "metadata": (
                result_frame.metadata
                if result_frame
                else {}
            ),
        }
    )


# ============================================================================
# REST: TEXT
# ============================================================================

@app.post("/api/text")
async def process_text(
    payload: Dict[str, str],
) -> JSONResponse:

    text = payload.get(
        "text",
        "",
    )

    if not text:

        raise HTTPException(
            status_code=400,
            detail="Missing text payload",
        )

    frame = TextFrame(
        payload=text
    )

    task = await runner.submit(
        frame
    )

    await runner.queue.join()

    task_result = runner.tasks[
        task.task_id
    ]

    result_frame = _assert_task_success(
        task_result
    )

    response_text = getattr(
        result_frame,
        "payload",
        "",
    )

    if isinstance(
        response_text,
        str,
    ):

        response_text = (
            response_text.strip()
        )

    if not response_text:

        response_text = (
            result_frame.metadata.get(
                "response_text"
            )
            or result_frame.metadata.get(
                "generated_text"
            )
            or result_frame.metadata.get(
                "text"
            )
            or (
                "Sorry, I couldn't generate "
                "a response at the moment."
            )
        )

        if isinstance(
            response_text,
            str,
        ):

            response_text = (
                response_text.strip()
            )

        if not response_text:

            response_text = (
                "Sorry, I couldn't generate "
                "a response at the moment."
            )

    return JSONResponse(
        {
            "task_id": task.task_id,
            "response": response_text,
            "metadata": (
                result_frame.metadata
                if result_frame
                else {}
            ),
        }
    )


# ============================================================================
# WEBSOCKET: VIDEO
# ============================================================================

@app.websocket("/ws/video")
async def ws_video(
    websocket: WebSocket,
) -> None:
    """
    Browser → server video stream.

    The browser sends JPEG frames approximately every
    300-500 ms.

    Every frame is analyzed for facial emotion.

    The latest facial result is stored in StreamingState.

    The background 2-second analyzer consumes this state
    automatically.
    """

    await websocket.accept()

    _video_clients.add(
        websocket
    )

    logger.info(
        "WebSocket /ws/video connected"
    )

    try:

        while True:

            frame_bytes = (
                await websocket.receive_bytes()
            )

            if not frame_bytes:
                continue

            # ----------------------------------------------------
            # FACE EMOTION
            # ----------------------------------------------------

            try:

                emotion_label, confidence = (
                    emotion_service.analyze_face_emotion(
                        frame_bytes
                    )
                )

                face_emotion = {
                    "emotion": emotion_label,
                    "confidence": round(
                        float(
                            confidence
                        ),
                        4,
                    ),
                }

                logger.info(
                    "[MEDIA] Video frame received | "
                    "size=%d bytes | "
                    "face_emotion=%s",
                    len(frame_bytes),
                    face_emotion,
                )

            except Exception as exc:

                logger.warning(
                    "Face emotion analysis failed: %s",
                    exc,
                )

                face_emotion = {
                    "emotion": "neutral",
                    "confidence": 0.0,
                }

            # ----------------------------------------------------
            # UPDATE SHARED STATE
            # ----------------------------------------------------

            async with _stream_lock:

                _stream_state.face_emotion = (
                    face_emotion
                )

            # ----------------------------------------------------
            # SEND CURRENT FACE RESULT
            # ----------------------------------------------------

            await _safe_send_json(
                websocket,
                {
                    "type": "face_emotion",
                    "face_emotion": face_emotion,
                },
            )

            # The 2-second background loop will independently
            # produce the fused emotion.

    except WebSocketDisconnect:

        logger.info(
            "WebSocket /ws/video disconnected"
        )

    except Exception as exc:

        logger.error(
            "WebSocket /ws/video error: %s",
            exc,
            exc_info=True,
        )

        try:

            await websocket.close(
                code=1011
            )

        except Exception:

            pass

    finally:

        _video_clients.discard(
            websocket
        )


# ============================================================================
# WEBSOCKET: AUDIO
# ============================================================================

@app.websocket("/ws/audio")
async def ws_audio(
    websocket: WebSocket,
) -> None:
    """
    Browser → server audio stream.

    The browser sends small MediaRecorder chunks over this socket.

    Previously each chunk was POSTed independently to Deepgram's
    one-shot REST endpoint (`deepgram_service.transcribe()`). That
    only works if every payload is a *complete, self-contained*
    audio file — but a MediaRecorder chunk stream only has valid
    container headers on the very first chunk, so every call after
    the first was headerless and Deepgram correctly rejected it as
    corrupt (400).

    Fix: open ONE persistent Deepgram WebSocket for the life of this
    connection (`start_stream()`), forward every chunk onto it as it
    arrives (`send_audio()`), and consume results from a background
    task via `transcripts()`. The persistent connection keeps the
    demuxer state across chunks, which is what actually lets a
    chunked MediaRecorder stream decode correctly.

    Speech emotion is still calculated from the accumulated
    transcript window every 2 seconds elsewhere in the app — this
    handler only changes how transcripts get produced.
    """

    await websocket.accept()

    _audio_clients.add(
        websocket
    )

    logger.info(
        "WebSocket /ws/audio connected"
    )

    deepgram_service = (
        DeepgramService()
    )

    try:
        await deepgram_service.start_stream()
    except Exception as exc:
        logger.error(
            "Failed to open Deepgram stream: %s",
            exc,
            exc_info=True,
        )
        await websocket.close(code=1011)
        _audio_clients.discard(websocket)
        return

    async def _consume_transcripts() -> None:
        """Background task: drain Deepgram results onto the socket
        as they arrive, independent of the audio-receive loop below."""
        try:
            async for result in deepgram_service.transcripts():

                transcript = deepgram_service.parse_transcript(result)

                if not transcript.strip():
                    continue

                is_final = bool(result.get("is_final"))

                logger.info(
                    "[STT] Transcript received (final=%s): %s",
                    is_final,
                    transcript,
                )

                if is_final:
                    async with _stream_lock:
                        _stream_state.transcript = transcript
                        _stream_state.transcript_chunks.append(transcript)

                await _safe_send_json(
                    websocket,
                    {
                        "type": "transcript",
                        "transcript": transcript,
                        "is_final": is_final,
                    },
                )
        except Exception as exc:
            logger.error(
                "Deepgram transcript consumer error: %s",
                exc,
                exc_info=True,
            )

    consumer_task = asyncio.create_task(_consume_transcripts())

    try:

        while True:

            audio_bytes = (
                await websocket.receive_bytes()
            )

            if not audio_bytes:
                continue

            logger.info(
                "[MEDIA] Audio chunk received | size=%d bytes",
                len(audio_bytes),
            )

            # ----------------------------------------------------
            # FORWARD ONTO THE PERSISTENT DEEPGRAM STREAM
            # ----------------------------------------------------

            try:
                await deepgram_service.send_audio(audio_bytes)
            except Exception as exc:
                logger.warning(
                    "Failed to forward audio to Deepgram: %s",
                    exc,
                )

    except WebSocketDisconnect:

        logger.info(
            "WebSocket /ws/audio disconnected"
        )

    except Exception as exc:

        logger.error(
            "WebSocket /ws/audio error: %s",
            exc,
            exc_info=True,
        )

        try:

            await websocket.close(
                code=1011
            )

        except Exception:

            pass

    finally:

        consumer_task.cancel()
        try:
            await consumer_task
        except (asyncio.CancelledError, Exception):
            pass

        await deepgram_service.close_stream()

        _audio_clients.discard(
            websocket
        )


# ============================================================================
# REST: MULTIMODAL
# ============================================================================

@app.post("/api/multimodal")
async def process_multimodal(
    request: Request,
) -> JSONResponse:
    """
    Manual multimodal request.

    Uses:

        face   = 0.55
        text   = 0.30
        speech = 0.15

    This endpoint is separate from the automatic live
    2-second face + speech analyzer.
    """

    global emotion_service

    if emotion_service is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Emotion service is unavailable"
            ),
        )

    content_type = request.headers.get(
        "content-type",
        "",
    )

    if not content_type.startswith(
        "multipart/form-data"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "/api/multimodal requires "
                "multipart/form-data"
            ),
        )

    # ------------------------------------------------------------
    # PARSE FORM
    # ------------------------------------------------------------

    try:

        form = await request.form()

        text = (
            form.get("text")
            or ""
        ).strip()

        image_upload = form.get(
            "image"
        )

        audio_upload = form.get(
            "audio"
        )

        image_bytes = (
            await image_upload.read()
            if image_upload is not None
            else b""
        )

        audio_bytes = (
            await audio_upload.read()
            if audio_upload is not None
            else b""
        )

    except AssertionError:

        raise HTTPException(
            status_code=400,
            detail=(
                "python-multipart is not installed. "
                "Send multipart/form-data with "
                "image/audio/text fields."
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not parse multipart "
                f"form data: {exc}"
            ),
        )

    if not text:

        raise HTTPException(
            status_code=400,
            detail="text field is required.",
        )

    # ------------------------------------------------------------
    # FACE
    # ------------------------------------------------------------

    if image_bytes:

        face_label, face_conf = (
            emotion_service.analyze_face_emotion(
                image_bytes
            )
        )

        face_emotion = {
            "emotion": face_label,
            "confidence": float(
                face_conf
            ),
        }

    else:

        async with _stream_lock:

            face_emotion = dict(
                _stream_state.face_emotion
            )

    # ------------------------------------------------------------
    # SPEECH
    # ------------------------------------------------------------

    if audio_bytes:

        deepgram_service = (
            DeepgramService()
        )

        transcript_payload = (
            await deepgram_service.transcribe(
                audio_bytes
            )
        )

        transcript = (
            deepgram_service.parse_transcript(
                transcript_payload
            )
        )

        if transcript.strip():

            speech_label, speech_conf = (
                emotion_service.classify_text_emotion(
                    transcript
                )
            )

        else:

            speech_label, speech_conf = (
                "neutral",
                0.0,
            )

        speech_sentiment = {
            "emotion": speech_label,
            "confidence": float(
                speech_conf
            ),
        }

    else:

        async with _stream_lock:

            transcript = (
                _stream_state.transcript
            )

            speech_sentiment = dict(
                _stream_state.speech_emotion
            )

    # ------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------

    text_label, text_conf = (
        emotion_service.classify_text_emotion(
            text
        )
    )

    text_emotion = {
        "emotion": text_label,
        "confidence": float(
            text_conf
        ),
    }

    # ------------------------------------------------------------
    # FULL THREE-MODALITY FUSION
    # ------------------------------------------------------------

    scores = {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "neutral": 0.0,
    }

    modalities = [
        (
            face_emotion,
            0.55,
        ),
        (
            text_emotion,
            0.30,
        ),
        (
            speech_sentiment,
            0.15,
        ),
    ]

    for emotion_data, weight in modalities:

        emotion = _normalize_emotion(
            emotion_data.get(
                "emotion",
                "neutral",
            )
        )

        confidence = _normalize_confidence(
            emotion_data.get(
                "confidence",
                0.0,
            )
        )

        if emotion in scores:

            scores[emotion] += (
                confidence *
                weight
            )

    dominant_emotion = max(
        scores,
        key=scores.get,
    )

    fusion = {
        "emotion": dominant_emotion,
        "confidence": round(
            scores[
                dominant_emotion
            ],
            4,
        ),
        "scores": {
            key: round(
                value,
                4,
            )
            for key, value in scores.items()
        },
    }

    # ------------------------------------------------------------
    # CREATE PIPELINE FRAME
    # ------------------------------------------------------------

    multimodal_frame = TextFrame(
        payload=text,
        metadata={
            "source": "multimodal",

            "face_emotion": face_emotion,

            "speech_sentiment": speech_sentiment,

            "text_emotion": text_emotion,

            "transcript": transcript,

            "fused_emotion": fusion,
        },
    )

    # ------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------

    task = await runner.submit(
        multimodal_frame
    )

    await runner.queue.join()

    task_result = runner.tasks[
        task.task_id
    ]

    result_frame = _assert_task_success(
        task_result
    )

    # ------------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------------

    response_text = getattr(
        result_frame,
        "payload",
        "",
    )

    if isinstance(
        response_text,
        str,
    ):

        response_text = (
            response_text.strip()
        )

    if not response_text:

        response_text = (
            result_frame.metadata.get(
                "response_text"
            )
            or result_frame.metadata.get(
                "generated_text"
            )
            or result_frame.metadata.get(
                "text"
            )
            or (
                "Sorry, I couldn't generate "
                "a response at the moment."
            )
        )

        if isinstance(
            response_text,
            str,
        ):

            response_text = (
                response_text.strip()
            )

    return JSONResponse(
        {
            "task_id": task.task_id,

            "response": response_text,

            "transcript": transcript,

            "metadata": (
                result_frame.metadata
                if result_frame
                else {}
            ),
        }
    )