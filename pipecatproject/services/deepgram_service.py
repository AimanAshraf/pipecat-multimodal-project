import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, Optional

import httpx
import websockets

from config import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class DeepgramService:
    """
    Deepgram Speech-to-Text service.

    Supports two modes:

    1. REST transcription
       transcribe(audio_bytes)

       Used for one-shot audio transcription and compatibility
       with the existing DeepgramProcessor.

    2. Live WebSocket transcription
       start_stream()
       send_audio()
       receive_results()
       close_stream()

       Used by the live browser audio stream.

    Live architecture:

        Browser microphone
              |
              | small audio chunks
              v
        FastAPI WebSocket
              |
              v
        Deepgram WebSocket
              |
              v
        incremental transcript
              |
              +----> UI in real time
              |
              +----> 2-second emotion window
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------
        #
        # First try the explicitly supplied key.
        # Then try settings.
        # Finally try the OS environment directly.
        #
        # The final fallback is important because your logs
        # currently show that settings.DE... is not resolving
        # even though the key exists in your .env file.
        # ----------------------------------------------------

        self.api_key = (
            api_key
            or getattr(settings, "DEEPGRAM_API_KEY", None)
            or os.getenv("DEEPGRAM_API_KEY")
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        self.model = (
            model
            or getattr(settings, "DEEPGRAM_MODEL", None)
            or os.getenv("DEEPGRAM_MODEL")
            or "nova-2"
        )

        # ----------------------------------------------------
        # LANGUAGE
        # ----------------------------------------------------

        self.language = (
            language
            or getattr(settings, "DEEPGRAM_LANGUAGE", None)
            or os.getenv("DEEPGRAM_LANGUAGE")
            or "en-US"
        )

        # ----------------------------------------------------
        # ENDPOINTS
        # ----------------------------------------------------

        self.rest_url = (
            "https://api.deepgram.com/v1/listen"
        )

        self.websocket_url = (
            "wss://api.deepgram.com/v1/listen"
        )

        # ----------------------------------------------------
        # LIVE CONNECTION
        # ----------------------------------------------------

        self.websocket = None

        self._stream_task: Optional[asyncio.Task] = None

        self._result_queue: asyncio.Queue = (
            asyncio.Queue()
        )

        self._stream_running = False

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        if not self.api_key:
            logger.warning(
                "DEEPGRAM_API_KEY is not configured"
            )
        else:
            logger.info(
                "Deepgram API key detected"
            )

        logger.info(
            "DeepgramService initialized "
            "model=%s language=%s",
            self.model,
            self.language,
        )

    # ========================================================
    # REST TRANSCRIPTION
    # ========================================================

    async def transcribe(
        self,
        audio_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Transcribe a complete audio payload using Deepgram REST.

        This method is retained for compatibility with the
        existing DeepgramProcessor.

        For true live transcription use start_stream().
        """

        if not audio_bytes:
            return {}

        if not self.api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not configured. "
                "Check your .env file and environment loading."
            )

        headers = {
            "Authorization": (
                f"Token {self.api_key}"
            ),
            "Content-Type": (
                self._detect_content_type(
                    audio_bytes
                )
            ),
        }

        params = {
            "model": self.model,
            "language": self.language,
            "smart_format": "true",
            "punctuate": "true",
        }

        logger.debug(
            "Sending %d bytes to Deepgram REST",
            len(audio_bytes),
        )

        try:

            async with httpx.AsyncClient(
                timeout=30.0
            ) as client:

                response = await client.post(
                    self.rest_url,
                    headers=headers,
                    params=params,
                    content=audio_bytes,
                )

            response.raise_for_status()

            result = response.json()

            logger.debug(
                "Deepgram REST transcription completed"
            )

            return result

        except httpx.HTTPStatusError as exc:

            logger.error(
                "Deepgram HTTP error: status=%s body=%s",
                exc.response.status_code,
                exc.response.text,
            )

            raise

        except httpx.RequestError as exc:

            logger.error(
                "Deepgram network error: %s",
                exc,
            )

            raise

        except Exception as exc:

            logger.error(
                "Deepgram transcription failed: %s",
                exc,
                exc_info=True,
            )

            raise

    # ========================================================
    # LIVE STREAM
    # ========================================================

    async def start_stream(self) -> None:
        """
        Open a persistent Deepgram WebSocket connection.

        This connection remains open while the user is speaking.

        Browser audio chunks are sent using send_audio().
        Deepgram results are collected by the background
        receiver task.
        """

        if not self.api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not configured."
            )

        # Avoid opening the same connection twice.
        if self.websocket is not None:
            logger.warning(
                "Deepgram stream already exists"
            )
            return

        params = {
            "model": self.model,
            "language": self.language,
            "smart_format": "true",
            "punctuate": "true",
            "interim_results": "true",
            "endpointing": "300",
            "vad_events": "true",
        }

        query_string = "&".join(
            f"{key}={value}"
            for key, value in params.items()
        )

        url = (
            f"{self.websocket_url}"
            f"?{query_string}"
        )

        headers = {
            "Authorization": (
                f"Token {self.api_key}"
            )
        }

        logger.info(
            "Opening Deepgram live WebSocket"
        )

        try:

            self.websocket = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            )

            self._stream_running = True

            logger.info(
                "Deepgram live WebSocket connected"
            )

            self._stream_task = asyncio.create_task(
                self._receive_loop()
            )

        except Exception as exc:

            self.websocket = None
            self._stream_running = False

            logger.error(
                "Failed to connect to Deepgram WebSocket: %s",
                exc,
                exc_info=True,
            )

            raise

    # ========================================================
    # SEND AUDIO
    # ========================================================

    async def send_audio(
        self,
        audio_bytes: bytes,
    ) -> None:
        """
        Send an audio chunk to the existing Deepgram
        WebSocket connection.

        This should be called continuously as audio arrives
        from the browser.
        """

        if not audio_bytes:
            return

        if self.websocket is None:
            logger.warning(
                "Cannot send audio: "
                "Deepgram stream is not connected"
            )
            return

        try:

            await self.websocket.send(
                audio_bytes
            )

        except Exception as exc:

            logger.error(
                "Failed to send audio to Deepgram: %s",
                exc,
                exc_info=True,
            )

            raise

    # ========================================================
    # RECEIVE LOOP
    # ========================================================

    async def _receive_loop(self) -> None:
        """
        Continuously receive messages from Deepgram.

        Results are placed into an asyncio.Queue so that
        the application can consume them asynchronously.
        """

        if self.websocket is None:
            return

        try:

            async for message in self.websocket:

                # --------------------------------------------
                # Deepgram normally sends JSON messages.
                # --------------------------------------------

                if isinstance(message, bytes):

                    logger.debug(
                        "Received binary message "
                        "from Deepgram: %d bytes",
                        len(message),
                    )

                    continue

                try:

                    data = json.loads(message)

                except json.JSONDecodeError:

                    logger.warning(
                        "Received invalid JSON from Deepgram"
                    )

                    continue

                # --------------------------------------------
                # Put every message into the queue.
                # --------------------------------------------

                await self._result_queue.put(
                    data
                )

                message_type = data.get(
                    "type"
                )

                # --------------------------------------------
                # Results
                # --------------------------------------------

                if message_type == "Results":

                    transcript = (
                        self._extract_transcript(
                            data
                        )
                    )

                    if transcript:

                        is_final = bool(
                            data.get(
                                "is_final",
                                False
                            )
                        )

                        speech_final = bool(
                            data.get(
                                "speech_final",
                                False
                            )
                        )

                        logger.info(
                            "Deepgram transcript | "
                            "final=%s speech_final=%s | %s",
                            is_final,
                            speech_final,
                            transcript,
                        )

                # --------------------------------------------
                # Speech started
                # --------------------------------------------

                elif message_type == "SpeechStarted":

                    logger.debug(
                        "Deepgram detected speech start"
                    )

                # --------------------------------------------
                # Utterance end
                # --------------------------------------------

                elif message_type == "UtteranceEnd":

                    logger.debug(
                        "Deepgram detected utterance end"
                    )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                elif message_type == "Metadata":

                    logger.debug(
                        "Deepgram stream metadata received"
                    )

        except Exception as exc:

            if self._stream_running:

                logger.error(
                    "Deepgram receive loop failed: %s",
                    exc,
                    exc_info=True,
                )

        finally:

            self._stream_running = False

            logger.info(
                "Deepgram receive loop stopped"
            )

    # ========================================================
    # GET NEXT RESULT
    # ========================================================

    async def receive_result(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next Deepgram WebSocket message.

        Returns None if timeout expires.
        """

        try:

            if timeout is None:

                return await self._result_queue.get()

            return await asyncio.wait_for(
                self._result_queue.get(),
                timeout=timeout,
            )

        except asyncio.TimeoutError:

            return None

    # ========================================================
    # TRANSCRIPT STREAM
    # ========================================================

    async def transcripts(
        self,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield Deepgram Results messages continuously.

        Useful for:

            async for result in service.transcripts():
                ...
        """

        while self._stream_running:

            result = await self.receive_result(
                timeout=1.0
            )

            if result is None:
                continue

            if result.get("type") != "Results":
                continue

            yield result

    # ========================================================
    # CLOSE STREAM
    # ========================================================

    async def close_stream(self) -> None:
        """
        Gracefully close the Deepgram WebSocket.
        """

        self._stream_running = False

        websocket = self.websocket

        self.websocket = None

        if websocket is None:
            return

        logger.info(
            "Closing Deepgram live WebSocket"
        )

        try:

            await websocket.send(
                json.dumps(
                    {
                        "type": "Finalize"
                    }
                )
            )

        except Exception:

            pass

        try:

            await websocket.send(
                json.dumps(
                    {
                        "type": "CloseStream"
                    }
                )
            )

        except Exception:

            pass

        try:

            await websocket.close()

        except Exception:

            pass

        if self._stream_task:

            if (
                self._stream_task
                is not asyncio.current_task()
            ):

                try:

                    await asyncio.wait_for(
                        self._stream_task,
                        timeout=2.0,
                    )

                except (
                    asyncio.TimeoutError,
                    asyncio.CancelledError,
                ):

                    self._stream_task.cancel()

                except Exception:

                    pass

            self._stream_task = None

        logger.info(
            "Deepgram live WebSocket closed"
        )

    # ========================================================
    # PARSE TRANSCRIPT
    # ========================================================

    def parse_transcript(
        self,
        response: Optional[Dict[str, Any]],
    ) -> str:
        """
        Extract transcript from either:

        - REST Deepgram response
        - WebSocket Results response
        """

        return self._extract_transcript(
            response
        )

    # ========================================================
    # INTERNAL TRANSCRIPT EXTRACTION
    # ========================================================

    def _extract_transcript(
        self,
        response: Optional[Dict[str, Any]],
    ) -> str:
        """
        Extract transcript text from a Deepgram response.
        """

        if not response:
            return ""

        try:

            # ------------------------------------------------
            # REST response
            #
            # {
            #   "results": {
            #       "channels": [...]
            #   }
            # }
            #
            # WebSocket response
            #
            # {
            #   "type": "Results",
            #   "channel": {
            #       "alternatives": [...]
            #   }
            # }
            # ------------------------------------------------

            if response.get("type") == "Results":

                channel = response.get(
                    "channel",
                    {}
                )

            else:

                results = response.get(
                    "results",
                    {}
                )

                if not isinstance(
                    results,
                    dict,
                ):
                    return ""

                channels = results.get(
                    "channels",
                    [],
                )

                if not isinstance(
                    channels,
                    list,
                ) or not channels:

                    return ""

                channel = channels[0]

            if not isinstance(
                channel,
                dict,
            ):
                return ""

            alternatives = channel.get(
                "alternatives",
                [],
            )

            if not isinstance(
                alternatives,
                list,
            ) or not alternatives:

                return ""

            alternative = alternatives[0]

            if not isinstance(
                alternative,
                dict,
            ):
                return ""

            transcript = alternative.get(
                "transcript",
                "",
            )

            if transcript is None:
                return ""

            return str(
                transcript
            ).strip()

        except Exception as exc:

            logger.error(
                "Failed to parse Deepgram response: %s",
                exc,
                exc_info=True,
            )

            return ""

    # ========================================================
    # CONTENT TYPE
    # ========================================================

    def _detect_content_type(
        self,
        audio_bytes: bytes,
    ) -> str:
        """
        Detect the content type for REST transcription.
        """

        if not audio_bytes:
            return "audio/webm"

        # WebM / Matroska
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            return "audio/webm"

        # WAV / RIFF
        if audio_bytes[:4] == b"RIFF":
            return "audio/wav"

        # OGG
        if audio_bytes[:4] == b"OggS":
            return "audio/ogg"

        # FLAC
        if audio_bytes[:4] == b"fLaC":
            return "audio/flac"

        return "audio/webm"