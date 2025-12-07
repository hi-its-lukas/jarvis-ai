"""Application configuration utilities."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, ValidationError


class Settings(BaseModel):
    """Centralized runtime configuration loaded from environment variables."""

    app_name: str = "Jarvis Backend"
    environment: str = "development"
    ha_url: AnyHttpUrl
    ha_token: str
    ollama_url: AnyHttpUrl = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ha_timeout_seconds: int = 10
    llm_timeout_seconds: int = 60
    discovery_refresh_seconds: int = 300
    cache_max_entities: int = 200

    class Config:
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load environment variables, ensuring required values are present."""

    load_dotenv()
    data: dict[str, Any] = {
        "app_name": os.getenv("APP_NAME", "Jarvis Backend"),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "ha_url": os.getenv("HA_URL", "http://127.0.0.1:8123"),
        "ha_token": os.getenv("HA_TOKEN", ""),
        "ollama_url": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.1"),
        "ha_timeout_seconds": int(os.getenv("HA_TIMEOUT_SECONDS", "10")),
        "llm_timeout_seconds": int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        "discovery_refresh_seconds": int(
            os.getenv("DISCOVERY_REFRESH_SECONDS", "300")
        ),
        "cache_max_entities": int(os.getenv("CACHE_MAX_ENTITIES", "200")),
    }

    if not data["ha_token"]:
        raise RuntimeError("HA_TOKEN is required for Home Assistant authentication")

    try:
        return Settings(**data)
    except ValidationError as exc:  # pragma: no cover - validated during startup
        raise RuntimeError("Invalid environment configuration") from exc
