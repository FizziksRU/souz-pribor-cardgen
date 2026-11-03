"""Провайдер polza.ai: российский агрегатор моделей с API, совместимым с OpenAI Chat Completions.

Документация: https://polza.ai/docs. Настройки берутся из окружения (.env):
LLM_API_KEY и LLM_MODEL обязательны, LLM_REASONING и LLM_TEMPERATURE — по желанию.
Счётчик расхода у клиента не потокобезопасен: одному потоку — свой клиент.
"""

import logging
import math
import os
from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import httpx2
import openai
from openai.types.chat import ChatCompletion

from .llm import LLMConfigError, LLMError, Message

log = logging.getLogger(__name__)

BASE_URL = "https://polza.ai/api/v1"
# Дольше, чем держит запрос сам polza.ai (600 с): пусть первым сдастся сервер — его ошибка бесплатна,
# а при обрыве с нашей стороны списывается уже сгенерированная часть ответа
TIMEOUT = 620.0
MAX_TOKENS = 8000  # с запасом на рассуждения: обрезанный JSON всё равно не разобрать
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
STATUS_HINTS = {
    401: "Неверный или отозванный ключ: проверьте LLM_API_KEY",
    402: "Не хватает средств на балансе polza.ai, исчерпан лимит расходов или слишком много параллельных запросов",
    403: "Доступ запрещён: ключ без прав на модель или запрос отклонён модерацией",
    404: "Модель не найдена: проверьте LLM_MODEL по каталогу https://polza.ai/models",
}
NOT_SENT = (httpx2.ConnectError, httpx2.ConnectTimeout)  # соединения не было — повтор бесплатен
DETAIL_LIMIT = 300  # знаков из текста ошибки сервиса: вместо JSON бывает целая HTML-страница


@dataclass(frozen=True)
class Usage:
    """Сколько потрачено с начала работы клиента. cost_rub — сумма, которую списал polza.ai."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_rub: float = 0.0
    unpriced: int = 0  # запросы с неизвестной стоимостью: нет cost_rub или связь оборвалась после отправки

    def _combine(self, other: "Usage", sign: int) -> "Usage":
        return Usage(*(getattr(self, f.name) + sign * getattr(other, f.name) for f in fields(self)))

    def __add__(self, other: "Usage") -> "Usage":
        return self._combine(other, 1)

    def __sub__(self, other: "Usage") -> "Usage":
        return self._combine(other, -1)


def _retryable(status: int) -> bool:
    """Таймаут, лимит запросов и сбои на стороне сервиса — по документации polza.ai."""
    return status >= 500 or status in (408, 429)


def _rub(value: object) -> float | None:
    """cost_rub из usage; None — стоимость неизвестна (поля нет или значение непонятное)."""
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def _detail(body: object, fallback: str) -> str:
    """Текст ошибки polza.ai: {"error": {"code", "message"}}; SDK иногда отдаёт уже внутренний объект."""
    inner = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(inner, dict) and inner.get("message"):
        code = inner.get("code")
        text = f"{inner['message']} [{code}]" if code else str(inner["message"])
    elif isinstance(inner, str) and inner.strip():
        text = inner.strip()
    else:
        text = fallback
    return text[:DETAIL_LIMIT]


def _env_float(env: Mapping[str, str], name: str, low: float, high: float) -> float | None:
    raw = env.get(name, "").strip().replace(",", ".")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise LLMConfigError(f"{name}: ожидается число от {low} до {high}, получено «{raw}».") from None
    if not low <= value <= high:
        raise LLMConfigError(f"{name}: ожидается число от {low} до {high}, получено {value}.")
    return value


class PolzaClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        reasoning: str | None = None,
        temperature: float | None = None,
        http_client=None,
    ):
        if os.environ.get("OPENAI_CUSTOM_HEADERS"):
            # SDK openai сам добавляет эти заголовки к каждому запросу — они ушли бы на polza.ai
            raise LLMConfigError("Задана переменная OPENAI_CUSTOM_HEADERS: её заголовки ушли бы на polza.ai. Уберите её.")
        self.model = model
        self.reasoning = reasoning
        self.temperature = temperature
        self.usage = Usage()
        self._secret = api_key
        # max_retries=0: повторы ведёт конвейер, иначе паузы и платные запросы перемножились бы
        self._sdk = openai.OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=TIMEOUT,
            max_retries=0,
            http_client=http_client,
            # SDK подставляет OPENAI_ORG_ID и OPENAI_PROJECT_ID из окружения — polza.ai они ни к чему
            default_headers={"OpenAI-Organization": openai.Omit(), "OpenAI-Project": openai.Omit()},
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "PolzaClient":
        env = os.environ if env is None else env
        api_key = env.get("LLM_API_KEY", "").strip()
        model = env.get("LLM_MODEL", "").strip()
        missing = [name for name, value in (("LLM_API_KEY", api_key), ("LLM_MODEL", model)) if not value]
        if missing:
            raise LLMConfigError(
                f"Для polza.ai не задано: {', '.join(missing)} — впишите в .env (см. .env.example) "
                "или запустите с --dry-run."
            )
        reasoning = env.get("LLM_REASONING", "").strip().lower() or None
        if reasoning is not None and reasoning not in REASONING_EFFORTS:
            raise LLMConfigError(f"LLM_REASONING: допустимо {', '.join(REASONING_EFFORTS)}; получено «{reasoning}».")
        temperature = _env_float(env, "LLM_TEMPERATURE", 0.0, 2.0)
        return cls(api_key, model, reasoning=reasoning, temperature=temperature)

    def _mask(self, text: str) -> str:
        """Сервис может процитировать ключ в тексте ошибки, а текст уходит в карточку и в веб-ответ."""
        return text.replace(self._secret, "***") if self._secret else text

    def _request(self, system: str, messages: Sequence[Message]) -> dict:
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }
        # Новые модели OpenAI и Anthropic не принимают temperature — передаём, только если задано
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning is not None:
            request["extra_body"] = {"reasoning": {"effort": self.reasoning}}
        return request

    def _count(self, usage) -> None:
        cost = _rub(getattr(usage, "cost_rub", None))
        prompt = getattr(usage, "prompt_tokens", None) or 0
        completion = getattr(usage, "completion_tokens", None) or 0
        self.usage = self.usage + Usage(1, prompt, completion, cost or 0.0, int(cost is None))
        if cost is None:
            log.warning("polza.ai %s: в ответе нет стоимости — в итоге она не учтена", self.model)
        else:
            log.info("polza.ai %s: %d + %d токенов, %.2f ₽", self.model, prompt, completion, cost)

    def _connection_error(self, exc: openai.APIConnectionError) -> LLMError:
        reason = self._mask(str(exc.__cause__ or exc))
        if isinstance(exc.__cause__, NOT_SENT):
            return LLMError(f"Нет связи с polza.ai: {reason}", retryable=True)
        self.usage = self.usage + Usage(requests=1, unpriced=1)
        what = f"polza.ai не ответил за {TIMEOUT:.0f} с" if isinstance(exc, openai.APITimeoutError) else (
            f"Связь с polza.ai оборвалась: {reason}"
        )
        # Запрос уже ушёл: за обрыв polza.ai списывает сгенерированную часть, повтор заплатил бы ещё раз
        return LLMError(f"{what}. Повтора нет: часть ответа могла быть оплачена.")

    def _status_error(self, exc: openai.APIStatusError) -> LLMError:
        status = exc.status_code
        hint = STATUS_HINTS.get(status, "Ошибка polza.ai")
        if status == 404:
            hint = f"{hint} (сейчас «{self.model}»)"
        detail = self._mask(_detail(exc.body, exc.message))
        return LLMError(f"{hint}. HTTP {status}: {detail}", retryable=_retryable(status))

    def _answer(self, response: object) -> str:
        if not isinstance(response, ChatCompletion):
            # SDK отдаёт строку, если на 200 пришёл не JSON: прокси, страница обслуживания
            raise LLMError("polza.ai ответил не в формате API (прокси или страница обслуживания?).", retryable=True)
        self._count(response.usage)
        error = getattr(response, "error", None)
        if error:
            detail = self._mask(_detail({"error": error}, str(error)))
            raise LLMError(f"polza.ai сообщил об ошибке модели: {detail}", retryable=True)
        if not response.choices:
            raise LLMError("polza.ai вернул ответ без вариантов.", retryable=True)
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"Ответ модели обрезан на {MAX_TOKENS} токенах — JSON неполный.")
        if choice.finish_reason == "content_filter":
            raise LLMError("Ответ модели заблокирован фильтром провайдера.")
        content = choice.message.content if choice.message is not None else None
        if choice.finish_reason == "error" or not content or not content.strip():
            raise LLMError("Модель вернула пустой ответ или оборвала его с ошибкой.", retryable=True)
        return content

    def complete(self, system: str, messages: Sequence[Message]) -> str:
        try:
            response = self._sdk.chat.completions.create(**self._request(system, messages))
        except openai.APIConnectionError as exc:
            raise self._connection_error(exc) from exc
        except openai.APIStatusError as exc:
            raise self._status_error(exc) from exc
        return self._answer(response)
