"""OpenAI-compatible API endpoints for Home Assistant voice integration."""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.openai_ha_adapter import generate_ha_response


router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    function_call: Optional[dict] = None


class FunctionDefinition(BaseModel):
    name: str
    parameters: dict = Field(default_factory=dict)


class EntityDefinition(BaseModel):
    entity_id: str
    state: str
    name: str
    attributes: dict | None = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[FunctionDefinition]] = None
    entities: List[EntityDefinition] = Field(default_factory=list)


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest) -> Any:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    response = await generate_ha_response(
        [message.model_dump() for message in payload.messages],
        [function.model_dump() for function in payload.functions] if payload.functions else None,
        [entity.model_dump() for entity in payload.entities],
    )

    return response

