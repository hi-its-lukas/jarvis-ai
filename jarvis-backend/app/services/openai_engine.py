"""OpenAI-compatible chat completion engine backed by the central Processor."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from app.logic.processor import Processor


class OpenAICompatibleEngine:
    def __init__(self, processor: Processor) -> None:
        self._processor = processor

    async def generate_chat_completion(
        self,
        *,
        messages: List[dict],
        functions: Optional[List[dict]] = None,
        function_call: Optional[str | dict] = "auto",
        entities: Optional[List[dict]] = None,
        model: str | None = None,
    ) -> Dict[str, Any]:

        # 1. User Input Validierung
        if not messages:
            raise ValueError("No messages provided")

        user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if not user_message:
            return self._build_text_response("Ich habe nichts gehört.")

        content = str(user_message.get("content", ""))

        # 2. Delegation an den Processor (Single Source of Truth)
        try:
            # Der Processor entscheidet selbstständig ob Fast/Smart Model und führt die Aktion aus.
            result_data = await self._processor.process(content)

            # 3. Response Handling
            # Da der Processor die Aktion bereits ausgeführt hat (via ha_api), bestätigen wir nur.
            service = result_data.get("service")
            result_val = result_data.get("result")

            # Falls der Processor ein direktes Ergebnis (z.B. Temperatur-Wert) zurückgab:
            if result_val and isinstance(result_val, (str, int, float)) and not isinstance(result_val, bool):
                response_text = str(result_val)
            elif service:
                entity = result_data.get("entity_id", "dem Gerät")
                response_text = f"Ich habe {service} für {entity} ausgeführt."
            else:
                response_text = "Befehl verarbeitet."

            return self._build_text_response(response_text, model=model)

        except RuntimeError as e:
            # Fehler vom Processor (z.B. keine Entity gefunden) fangen wir sauber ab
            return self._build_text_response(f"Das konnte ich nicht ausführen: {str(e)}", model=model)
        except Exception:
            return self._build_text_response("Es ist ein unbekannter Fehler aufgetreten.", model=model)

    def _build_text_response(self, text: str, model: str | None = None) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "jarvis",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
