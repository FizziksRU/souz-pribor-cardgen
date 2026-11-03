import json
from pathlib import Path

import pytest

from cardgen.llm import LLMError

ROOT = Path(__file__).resolve().parent.parent
VHOD = ROOT / "vhod"
PD100 = VHOD / "pd100-di1-6-111-0-5.json"
TM510 = VHOD / "tm-510r-0-2-5mpa-g1-2.json"

FILLER = (
    "Прибор предназначен для постоянного контроля давления в технологических линиях "
    "и инженерных сетях, показания читаются без дополнительного оборудования. "
)


def make_description(length: int = 900) -> str:
    """Текст без цифр нужной длины — проверке чисел он не мешает."""
    text = ""
    while len(text) < length:
        text += FILLER
    return text[:length].rstrip() + "."


def make_card(**overrides) -> dict:
    card = {
        "seo_title": "Датчик давления ОВЕН ПД100-ДИ1,6-111-0,5",
        "seo_description": make_description(),
        "search_queries": [
            "датчик давления ОВЕН ПД100",
            "ПД100-ДИ1,6-111-0,5 купить",
            "преобразователь избыточного давления 4–20 мА",
        ],
        "commercial": {
            "audience": ["службы водоснабжения", "котельные"],
            "tasks": ["контроль давления в трубопроводах"],
        },
        "data_issues": [],
    }
    card.update(overrides)
    return card


def card_json(**overrides) -> str:
    return json.dumps(make_card(**overrides), ensure_ascii=False)


class FakeLLM:
    """Отдаёт заранее заготовленные ответы по очереди и запоминает запросы."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, list(messages)))
        reply = self._responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def llm_error():
    return LLMError("Сервис LLM недоступен")
