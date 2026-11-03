"""Граница с LLM: общий интерфейс клиента и ошибки.

Конкретный провайдер подключается отдельным классом с методом complete()
(сейчас — polza.ai, см. polza.py); остальной код зависит только от этого интерфейса.
"""

import os
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


class LLMError(Exception):
    """Ошибка обращения к модели. retryable — имеет ли смысл повторить позже."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMConfigError(LLMError):
    """Модель не настроена: нет провайдера, ключа или модели."""


class LLMClient(Protocol):
    def complete(self, system: str, messages: Sequence[Message]) -> str:
        """Возвращает текст ответа модели или бросает LLMError."""


def _polza() -> LLMClient:
    from .polza import PolzaClient  # SDK грузится, только когда модель действительно нужна

    return PolzaClient.from_env()


PROVIDERS: dict[str, Callable[[], LLMClient]] = {"polza": _polza}


def client_from_env() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        raise LLMConfigError(
            "Модель не настроена: задайте LLM_PROVIDER и ключ в .env (см. .env.example) "
            "или запустите с --dry-run, чтобы проверить всё, кроме вызова модели."
        )
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Неизвестный LLM_PROVIDER «{provider}». Доступные: {', '.join(sorted(PROVIDERS))}.")
    return PROVIDERS[provider]()
