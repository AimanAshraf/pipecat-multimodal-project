# Emotion-Aware Conversational AI Assistant using Pipecat

This project demonstrates a modular, production-ready conversational assistant built around a custom Pipecat pipeline. It integrates speech transcription, facial emotion recognition, text emotion classification, emotion fusion, contextual prompt engineering, streaming LLM responses, and expressive TTS.

## Features

- Pipecat pipeline with custom frame processors
- Real-time audio and webcam frame ingestion
- Deepgram speech-to-text
- HuggingFace text emotion recognition
- FER facial emotion detection
- Rule-based emotion fusion
- Groq LLM prompt generation and streaming responses
- Cartesia text-to-speech voice styling
- Daily WebRTC integration for browser transport
- FastAPI backend with structured logging
- Conversation memory and emotion-aware prompts
- Unit tests for processors and pipeline functionality

## Folder Structure

- `app.py` - FastAPI application entrypoint
- `pipeline.py` - Pipecat pipeline assembly
- `config.py` - environment configuration and settings
- `pipecat/` - custom Pipecat framework implementation
- `processors/` - frame processor implementations
- `services/` - wrappers for Deepgram, Groq, Cartesia, Daily, and emotion models
- `models/` - domain model definitions
- `utils/` - logging and helper utilities
- `frontend/` - browser client assets for Daily and media capture
- `tests/` - unit and integration tests

## Installation

1. Copy `.env.example` to `.env` and fill in the API keys.
2. Create a Python 3.11+ virtual environment.
3. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` in a browser or serve it from the backend static files.

## Usage

1. Open the frontend in a browser.
2. Start the Daily WebRTC session.
3. Allow camera and microphone access.
4. Speak naturally and watch the assistant analyze mood, respond empathetically, and generate voice output.
