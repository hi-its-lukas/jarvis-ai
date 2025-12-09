import pytest
from app.services.openai_ha_adapter import generate_ha_response


ENTITIES = [
    {"entity_id": "light.wohnzimmer", "state": "on", "name": "Wohnzimmerlampe", "attributes": {}},
    {"entity_id": "light.bad", "state": "off", "name": "Deckenlampe Bad", "attributes": {}},
    {"entity_id": "cover.wohnzimmer", "state": "open", "name": "Rollo Wohnzimmer", "attributes": {}},
    {"entity_id": "sensor.schlafzimmer_temperature", "state": "21", "name": "Schlafzimmer Temperatur", "attributes": {}},
]


def _user_message(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


@pytest.mark.anyio
async def test_list_lights_on():
    response = await generate_ha_response(_user_message("Welche Lampen sind an?"), [], ENTITIES)
    choice = response["choices"][0]

    assert choice["finish_reason"] == "stop"
    assert "Wohnzimmerlampe" in choice["message"]["content"]


@pytest.mark.anyio
async def test_turn_on_bathroom_light():
    response = await generate_ha_response(_user_message("Schalte das Licht im Bad an"), [], ENTITIES)
    choice = response["choices"][0]
    call = choice["message"]["function_call"]

    assert choice["finish_reason"] == "function_call"
    assert call["name"] == "turn_on"
    assert "light.bad" in call["arguments"]


@pytest.mark.anyio
async def test_close_living_room_covers():
    response = await generate_ha_response(_user_message("Mach die Rollos im Wohnzimmer runter"), [], ENTITIES)
    choice = response["choices"][0]
    call = choice["message"]["function_call"]

    assert call["name"] == "close_cover"
    assert "cover.wohnzimmer" in call["arguments"]


@pytest.mark.anyio
async def test_temperature_answer():
    response = await generate_ha_response(_user_message("Wie warm ist es im Schlafzimmer?"), [], ENTITIES)
    choice = response["choices"][0]

    assert choice["finish_reason"] == "stop"
    assert "21" in choice["message"]["content"]
