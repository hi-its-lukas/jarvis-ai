"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import AsyncGenerator

from fastapi import FastAPI

from app.api import openai_api, routes
from app.core.config import get_settings
from app.logic.context import ConversationContext
from app.logic.processor import Processor
from app.services.discovery import DiscoveryService
from app.services.ha_api import HomeAssistantAPI
from app.services.ollama_api import OllamaAPI

logger = logging.getLogger("jarvis")
logging.basicConfig(level=logging.INFO)

settings = get_settings()
ha_api = HomeAssistantAPI(
    settings.ha_url, settings.ha_token, timeout_seconds=settings.ha_timeout_seconds
)
discovery = DiscoveryService(
    ha_api,
    cache_max_entities=settings.cache_max_entities,
)
ollama = OllamaAPI(
    settings.ollama_url,
    settings.ollama_model,
    timeout_seconds=settings.llm_timeout_seconds,
)
context = ConversationContext()
processor = Processor(
    ha_api=ha_api,
    discovery=discovery,
    ollama=ollama,
    context=context,
)


def _build_app() -> FastAPI:
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Starting Jarvis backend (%s)", settings.environment)
        await discovery.refresh()

        refresh_task = asyncio.create_task(_discovery_refresh_loop())
        app.state.processor = processor
        try:
            yield
        finally:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh_task

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.include_router(routes.router, prefix="/api")
    app.include_router(openai_api.router)
    return app


async def _discovery_refresh_loop() -> None:
    delay = max(30, settings.discovery_refresh_seconds)
    while True:
        await asyncio.sleep(delay)
        try:
            await discovery.refresh()
            logger.debug("Discovery cache refreshed")
        except Exception as exc:  # pragma: no cover
            logger.warning("Discovery refresh failed: %s", exc)


app = _build_app()
