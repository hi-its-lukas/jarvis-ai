"""OpenAI-compatible API endpoints for Jarvis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import base64
import time
import uuid

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from app.services.function_registry import FUNCTION_DEFINITIONS
from app.services.openai_engine import OpenAICompatibleEngine


router = APIRouter()


FILE_STORE: Dict[str, Dict[str, Any]] = {}
ASSISTANT_STORE: Dict[str, Dict[str, Any]] = {}
FINE_TUNE_JOBS: Dict[str, Dict[str, Any]] = {}


def _error_response(message: str, *, code: int = 400) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": code,
        }
    }


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


class EmbeddingInput(BaseModel):
    model: str
    input: Any


class ModerationRequest(BaseModel):
    model: Optional[str] = Field(default="omni-moderation-latest")
    input: Any


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = "gpt-image-1"
    prompt: str
    size: Optional[str] = "512x512"
    response_format: Optional[str] = Field(default="url", pattern="^(url|b64_json)$")


class FileUploadResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str = "processed"


class AssistantRequest(BaseModel):
    name: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None


class FineTuneRequest(BaseModel):
    training_file: str
    model: Optional[str] = None
    suffix: Optional[str] = None


def _get_engine(request: Request) -> OpenAICompatibleEngine:
    engine = getattr(request.app.state, "openai_engine", None)
    if engine is None:
        # Hole den zentralen Processor aus dem State (wird in main.py initialisiert)
        processor = getattr(request.app.state, "processor", None)
        if not processor:
            raise RuntimeError("Global Processor not initialized")

        # Inject Processor only - keine direkten Services mehr!
        engine = OpenAICompatibleEngine(processor=processor)
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

    completion_id = completion.get("id", f"resp-{uuid.uuid4()}")
    completion_id = completion_id.replace("chatcmpl-", "resp-", 1).replace("cmpl-", "resp-", 1)

    return ResponseObject(
        id=completion_id,
        model=completion.get("model"),
        created=completion.get("created"),
        output=output_blocks,
    )


def _fake_embedding_vector(text: str, length: int = 8) -> List[float]:
    if not text:
        return [0.0 for _ in range(length)]
    seed = sum(ord(char) for char in text)
    return [((seed + idx * 31) % 997) / 997 for idx in range(length)]


def _store_file(upload: UploadFile, content: bytes, purpose: str) -> Dict[str, Any]:
    file_id = f"file-{uuid.uuid4()}"
    metadata = {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": upload.filename or "unknown",
        "purpose": purpose,
        "status": "processed",
    }
    FILE_STORE[file_id] = {"metadata": metadata, "content": content}
    return metadata


def _assistant_object(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": data["id"],
        "object": "assistant",
        "created_at": data["created_at"],
        "name": data.get("name"),
        "model": data.get("model"),
        "instructions": data.get("instructions"),
        "tools": data.get("tools", []),
        "metadata": data.get("metadata", {}),
    }


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

    return _build_chat_completion_response(response, fallback_model=payload.model)


@router.post("/v1/completions")
async def text_completions(request: Request, payload: Dict[str, Any]) -> Any:
    prompt = payload.get("prompt")
    model = payload.get("model")
    if prompt is None:
        raise HTTPException(status_code=400, detail=_error_response("Prompt is required"))

    messages = [{"role": "user", "content": _normalize_content(prompt)}]
    engine = _get_engine(request)

    try:
        completion = await engine.generate_chat_completion(
            messages=messages,
            functions=FUNCTION_DEFINITIONS,
            function_call="auto",
            entities=[],
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_error_response(str(exc))) from exc

    return _build_chat_completion_response(completion, fallback_model=model)


@router.post("/v1/completions")
async def text_completions(request: Request, payload: Dict[str, Any]) -> Any:
    prompt = payload.get("prompt")
    model = payload.get("model")
    if prompt is None:
        raise HTTPException(status_code=400, detail=_error_response("Prompt is required"))

    messages = [{"role": "user", "content": _normalize_content(prompt)}]
    engine = _get_engine(request)

    try:
        completion = await engine.generate_chat_completion(
            messages=messages,
            functions=FUNCTION_DEFINITIONS,
            function_call="auto",
            entities=[],
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_error_response(str(exc))) from exc

    return completion


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


@router.get("/v1/models")
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


@router.post("/v1/embeddings")
async def create_embedding(payload: EmbeddingInput) -> Dict[str, Any]:
    normalized_input = _normalize_content(payload.input)
    embedding = _fake_embedding_vector(normalized_input)
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": 0,
            }
        ],
        "model": payload.model,
        "usage": {"prompt_tokens": len(normalized_input.split()), "total_tokens": len(normalized_input.split())},
    }


@router.post("/v1/moderations")
async def create_moderation(payload: ModerationRequest) -> Dict[str, Any]:
    return {
        "id": f"modr-{uuid.uuid4()}",
        "model": payload.model,
        "results": [
            {
                "flagged": False,
                "categories": {},
                "category_scores": {},
            }
        ],
    }


@router.post("/v1/images/generations")
async def generate_images(payload: ImageGenerationRequest) -> Dict[str, Any]:
    created = int(time.time())
    placeholder_bytes = f"Image for: {payload.prompt}".encode("utf-8")
    b64_placeholder = base64.b64encode(placeholder_bytes).decode("utf-8")
    data_entry: Dict[str, Any]
    if payload.response_format == "b64_json":
        data_entry = {"b64_json": b64_placeholder}
    else:
        data_entry = {"url": f"data:image/png;base64,{b64_placeholder}"}
    return {"created": created, "data": [data_entry]}


@router.get("/v1/files")
async def list_files() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [meta["metadata"] for meta in FILE_STORE.values()],
    }


@router.post("/v1/files")
async def upload_file(file: UploadFile = File(...), purpose: str = "assistants") -> FileUploadResponse:
    if not file:
        raise HTTPException(status_code=400, detail=_error_response("File is required"))
    content = await file.read()
    metadata = _store_file(file, content, purpose)
    return FileUploadResponse(**metadata)


@router.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str) -> Any:
    record = FILE_STORE.get(file_id)
    if not record:
        raise HTTPException(status_code=404, detail=_error_response("File not found", code=404))
    return record["metadata"]


@router.post("/v1/assistants")
async def create_assistant(payload: AssistantRequest) -> Dict[str, Any]:
    assistant_id = f"asst-{uuid.uuid4()}"
    created_at = int(time.time())
    data = {
        "id": assistant_id,
        "created_at": created_at,
        "name": payload.name,
        "model": payload.model,
        "instructions": payload.instructions,
        "tools": payload.tools or [],
        "metadata": {},
    }
    ASSISTANT_STORE[assistant_id] = data
    return _assistant_object(data)


@router.get("/v1/assistants")
async def list_assistants() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [_assistant_object(item) for item in ASSISTANT_STORE.values()],
    }


@router.get("/v1/assistants/{assistant_id}")
async def get_assistant(assistant_id: str) -> Dict[str, Any]:
    data = ASSISTANT_STORE.get(assistant_id)
    if not data:
        raise HTTPException(status_code=404, detail=_error_response("Assistant not found", code=404))
    return _assistant_object(data)


@router.post("/v1/fine_tuning/jobs")
async def create_fine_tune_job(payload: FineTuneRequest) -> Dict[str, Any]:
    job_id = f"ftjob-{uuid.uuid4()}"
    created_at = int(time.time())
    job = {
        "id": job_id,
        "object": "fine_tuning.job",
        "model": payload.model or "gpt-3.5-turbo",
        "created_at": created_at,
        "fine_tuned_model": None,
        "organization_id": "org-local",
        "status": "succeeded",
        "training_file": payload.training_file,
    }
    FINE_TUNE_JOBS[job_id] = job
    return job


@router.get("/v1/fine_tuning/jobs")
async def list_fine_tune_jobs() -> Dict[str, Any]:
    return {"object": "list", "data": list(FINE_TUNE_JOBS.values())}


@router.get("/v1/fine_tuning/jobs/{job_id}")
async def get_fine_tune_job(job_id: str) -> Dict[str, Any]:
    job = FINE_TUNE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=_error_response("Job not found", code=404))
    return job

