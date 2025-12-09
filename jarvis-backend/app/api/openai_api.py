"""OpenAI-compatible API endpoints for Jarvis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from app.services.entity_service import EntityService
from app.services.function_registry import FUNCTION_DEFINITIONS
from app.services.openai_engine import OpenAICompatibleEngine


router = APIRouter()


def _error_response(message: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error"}}


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[dict] = None


class FunctionDefinition(BaseModel):
    name: str
    parameters: dict = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[str | dict] = Field(default="auto")
    entities: List[dict] = Field(default_factory=list)


def _get_engine(request: Request) -> OpenAICompatibleEngine:
    engine = getattr(request.app.state, "openai_engine", None)
    if engine is None:
        discovery = getattr(request.app.state, "discovery", None)
        ollama = getattr(request.app.state, "ollama", None)
        entity_service = EntityService(discovery=discovery)
        engine = OpenAICompatibleEngine(
            discovery=discovery,
            ollama=ollama,
            entity_service=entity_service,
        )
        request.app.state.openai_engine = engine
    return engine


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, payload: ChatCompletionRequest) -> Any:
    if not payload.messages:
        raise HTTPException(status_code=400, detail=_error_response("No messages provided"))

    engine = _get_engine(request)

    try:
        response = await engine.generate_chat_completion(
            messages=[message.model_dump() for message in payload.messages],
            functions=[function.model_dump() for function in payload.functions]
            if payload.functions
            else FUNCTION_DEFINITIONS,
            function_call=payload.function_call,
            entities=payload.entities,
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_error_response(str(exc))) from exc

    return response


@router.post("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "jarvis-mistral", "object": "model"},
            {"id": "jarvis-hermes", "object": "model"},
        ],
    }


@router.post("/v1/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, str]:
    if not file:
        raise HTTPException(status_code=400, detail=_error_response("No file provided"))

    content = await file.read()
    placeholder = (
        "Transkription wurde entgegengenommen, ein STT-Dienst ist jedoch nicht konfiguriert."
    )
    inferred = placeholder if not content else placeholder + f" Dateigröße: {len(content)} Bytes."
    return {"text": inferred}


@router.post("/v1/audio/speech")
async def audio_speech(payload: Dict[str, Any]) -> Response:
    text_input = str(payload.get("input", ""))
    if not text_input:
        raise HTTPException(status_code=400, detail=_error_response("Input text is required"))

    audio_bytes = text_input.encode("utf-8")
    headers = {"Content-Type": "audio/wav"}
    return Response(content=audio_bytes, headers=headers, media_type="audio/wav")

