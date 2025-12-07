"""API routes for Jarvis backend."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.logic.processor import Processor

router = APIRouter()


class ProcessRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Natural language command")


class ProcessResponse(BaseModel):
    path: str
    service: str | None = None
    entity_id: str | None = None
    result: dict | list | str | int | float | bool | None = None


def get_processor(request: Request) -> Processor:
    processor: Processor | None = getattr(request.app.state, "processor", None)
    if not processor:
        raise RuntimeError("Processor has not been initialised")
    return processor


@router.post("/process", response_model=ProcessResponse)
async def process_text(
    payload: ProcessRequest,
    processor: Processor = Depends(get_processor),
) -> dict:
    try:
        result = await processor.process(payload.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - FastAPI handles
        raise HTTPException(status_code=500, detail="Internal error") from exc

    return {
        "path": result.get("path", "unknown"),
        "service": result.get("service"),
        "entity_id": result.get("entity_id"),
        "result": result.get("result"),
    }
