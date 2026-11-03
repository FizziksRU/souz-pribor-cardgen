"""Конвейер для одного товара: разбор входа -> модель -> проверка -> повтор -> результат."""

import logging
import time
from dataclasses import dataclass
from typing import Callable

from .analysis import MIN_DESCRIPTION, MIN_SPECS, Analysis, analyze
from .llm import LLMClient, LLMError, Message
from .prompt import SYSTEM_PROMPT, build_retry_message, build_user_message
from .schema import CardResult, CardText, ProductInput
from .validate import CardFormatError, check_card, parse_card

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # первый ответ и одна попытка исправить его по списку ошибок
RETRY_DELAYS = (2.0, 5.0)  # паузы перед повтором при временной ошибке сервиса (лимит, 5xx)
NOT_ENOUGH = (
    f"Недостаточно данных: нужно хотя бы {MIN_SPECS} характеристики или описание "
    f"от {MIN_DESCRIPTION} символов — иначе описание на 800+ символов не написать без домыслов."
)


@dataclass(frozen=True)
class _Attempt:
    card: CardText | None
    errors: tuple[str, ...]


def _base(product: ProductInput, analysis: Analysis) -> dict:
    return {
        "product_name": product.name,
        "kind": analysis.kind,
        "key_specs": list(analysis.key_specs),
        "missing_fields": list(analysis.missing),
        "needs_clarification": list(analysis.clarify),
    }


def _ask(client: LLMClient, messages: list[Message], sleep: Callable[[float], None]) -> str:
    """Один запрос к модели; временные ошибки повторяются, любые сбои клиента — LLMError."""
    delays = iter(RETRY_DELAYS)
    while True:
        try:
            raw = client.complete(SYSTEM_PROMPT, messages)
        except LLMError as exc:
            delay = next(delays, None)
            if not exc.retryable or delay is None:
                raise
            log.warning("Временная ошибка модели, повтор через %s с: %s", delay, exc)
            sleep(delay)
            continue
        except Exception as exc:  # клиент — внешний код: его сбой губит карточку, а не весь прогон
            log.exception("Сбой клиента модели")
            raise LLMError(f"Сбой клиента модели: {exc!r}") from exc
        if not isinstance(raw, str):
            raise LLMError(f"Клиент модели вернул {type(raw).__name__} вместо текста.")
        return raw


def _evaluate(raw: str, product: ProductInput, analysis: Analysis) -> _Attempt:
    try:
        card = parse_card(raw)
    except CardFormatError as exc:
        return _Attempt(card=None, errors=(str(exc),))
    return _Attempt(card=card, errors=check_card(card, product, analysis))


def _better(best: _Attempt | None, last: _Attempt) -> _Attempt | None:
    """Из разобранных ответов оставляем тот, где меньше замечаний (при равенстве — более поздний)."""
    if last.card is None:
        return best
    if best is None or len(last.errors) <= len(best.errors):
        return last
    return best


def _finish(base: dict, best: _Attempt | None, last: _Attempt | None, attempts: int, error: str | None) -> CardResult:
    if best is None:
        reason = error or "Модель не вернула JSON по формату: " + " ".join(last.errors if last else ())
        return CardResult(**base, status="error", attempts=attempts, error=reason)
    return CardResult(
        **base,
        status="needs_review",
        card=best.card,
        validation_errors=list(best.errors),
        attempts=attempts,
        error=error,
    )


def _run_model(client: LLMClient, product: ProductInput, analysis: Analysis, base: dict, sleep) -> CardResult:
    messages = [Message("user", build_user_message(product, analysis))]
    best: _Attempt | None = None
    last: _Attempt | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = _ask(client, messages, sleep)
        except LLMError as exc:
            return _finish(base, best, last, attempt - 1, str(exc))
        last = _evaluate(raw, product, analysis)
        if last.card is not None and not last.errors:
            return CardResult(**base, status="ok", card=last.card, attempts=attempt)
        best = _better(best, last)
        messages = [*messages, Message("assistant", raw), Message("user", build_retry_message(list(last.errors)))]
    return _finish(base, best, last, MAX_ATTEMPTS, None)


def generate_card(
    product: ProductInput,
    client: LLMClient | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> CardResult:
    """client=None — режим без модели: только разбор входа и собранный промпт."""
    analysis = analyze(product)
    base = _base(product, analysis)
    if not analysis.enough_data:
        return CardResult(**base, status="insufficient_data", error=NOT_ENOUGH)
    if client is None:
        prompt = f"{SYSTEM_PROMPT}\n----- сообщение пользователя -----\n{build_user_message(product, analysis)}"
        return CardResult(**base, status="dry_run", prompt=prompt)
    return _run_model(client, product, analysis, base, sleep)
