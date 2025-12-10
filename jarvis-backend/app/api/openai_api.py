"""OpenAI-compatible API endpoints for Jarvis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import uuid

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


class ResponseContentPart(BaseModel):
    type: str = Field(default="text")
    text: Optional[str] = None


class ResponseMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None


class ResponseRequest(BaseModel):
    model: str
    messages: Optional[List[ResponseMessage]] = None
    input: Optional[Any] = None
    stream: Optional[bool] = False


class ResponseMessageOutput(BaseModel):
    type: str = Field(default="message")
    role: str
    content: List[ResponseContentPart]


class ResponseObject(BaseModel):
    id: str
    object: str = Field(default="response")
    type: str = Field(default="response")
    model: Optional[str] = None
    created: Optional[int] = None
    output: List[ResponseMessageOutput]


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


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "input_text" and "input_text" in item:
                    parts.append(str(item.get("input_text", "")))
            else:
                parts.append(str(item))
        return " ".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        return " ".join(str(value) for value in content.values())
    return str(content) if content is not None else ""


def _build_response_output_from_completion(completion: Dict[str, Any]) -> ResponseObject:
    choices = completion.get("choices") or []
    output_blocks: List[ResponseMessageOutput] = []

    for choice in choices:
        message = choice.get("message") or {}
        role = message.get("role", "assistant")
        message_content = message.get("content")
        function_call = message.get("function_call")

        if not message_content and function_call:
            message_content = _normalize_content(function_call)

        text_content = _normalize_content(message_content)
        output_blocks.append(
            ResponseMessageOutput(
                role=role,
                content=[ResponseContentPart(text=text_content)],
            )
        )

    return ResponseObject(
        id=completion.get("id", f"resp-{uuid.uuid4()}").replace("cmpl-", "resp-", 1),
        model=completion.get("model"),
        created=completion.get("created"),
        output=output_blocks,
    )


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


@router.post("/v1/responses")
async def responses(request: Request, payload: ResponseRequest) -> Any:
    messages: List[dict] = []

    if payload.messages:
        messages = [
            {
                "role": message.role,
                "content": _normalize_content(message.content),
                "name": message.name,
            }
            for message in payload.messages
        ]
    elif payload.input is not None:
        normalized_input = _normalize_content(payload.input)
        if not normalized_input:
            raise HTTPException(
                status_code=400,
                detail=_error_response("Input is required when messages are not provided"),
            )
        messages = [{"role": "user", "content": normalized_input}]
    else:
        raise HTTPException(
            status_code=400, detail=_error_response("Either messages or input must be provided")
        )

    engine = _get_engine(request)

    try:
        completion = await engine.generate_chat_completion(
            messages=messages,
            functions=FUNCTION_DEFINITIONS,
            function_call="auto",
            entities=[],
            model=payload.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_error_response(str(exc))) from exc

    return _build_response_output_from_completion(completion)


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

