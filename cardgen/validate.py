"""Проверка ответа модели кодом: формат, длины, числа и упоминания против входных данных."""

import json
import re

from pydantic import ValidationError

from .analysis import Analysis
from .facts import extract_facts, unknown_facts
from .schema import CardText, ProductInput

TITLE_MAX = 70
DESCRIPTION_MIN, DESCRIPTION_MAX = 800, 1200
QUERIES_MIN, QUERIES_MAX = 3, 5

MARKETING = re.compile(
    r"(?<![а-яё])((?:наи)?лучш\w*|уникальн\w*|инновацион\w*|исключительн\w*|идеальн\w*|непревзойд\w*|революцион\w*)",
    re.IGNORECASE,
)


class CardFormatError(ValueError):
    """Ответ модели — не JSON-объект нужного формата."""


def parse_card(raw: str) -> CardText:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise CardFormatError("В ответе нет JSON-объекта.")
    try:
        data = json.loads(raw[start : end + 1])
        return CardText.model_validate(data)
    except json.JSONDecodeError as exc:
        raise CardFormatError(f"Ответ не разбирается как JSON: {exc.msg}.") from exc
    except ValidationError as exc:
        fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
        raise CardFormatError(f"JSON не по формату, ошибки в полях: {fields}.") from exc


def _input_text(product: ProductInput) -> str:
    specs = " ".join(f"{k} {v}" for k, v in product.specs.items())
    return " ".join(filter(None, (product.name, product.manufacturer, product.category, specs, product.description)))


def _text_parts(card: CardText) -> list[tuple[str, str]]:
    return [
        ("SEO-заголовок", card.seo_title),
        ("SEO-описание", card.seo_description),
        *(("поисковые запросы", q) for q in card.search_queries),
        *(("кому подходит", a) for a in card.commercial.audience),
        *(("задачи", t) for t in card.commercial.tasks),
    ]


def _check_shape(card: CardText) -> list[str]:
    errors = []
    title, description = card.seo_title.strip(), card.seo_description.strip()
    if not title or len(title) > TITLE_MAX:
        errors.append(f"SEO-заголовок: {len(title)} символов, нужно от 1 до {TITLE_MAX}.")
    if not DESCRIPTION_MIN <= len(description) <= DESCRIPTION_MAX:
        errors.append(f"SEO-описание: {len(description)} символов, нужно {DESCRIPTION_MIN}–{DESCRIPTION_MAX}.")
    queries = [q.strip().lower() for q in card.search_queries]
    if not QUERIES_MIN <= len(queries) <= QUERIES_MAX:
        errors.append(f"Поисковых запросов {len(queries)}, нужно {QUERIES_MIN}–{QUERIES_MAX}.")
    if len(set(queries)) != len(queries):
        errors.append("Поисковые запросы повторяются — нужны разные запросы.")
    for block, items in (("кому подходит", card.commercial.audience), ("какие задачи решает", card.commercial.tasks)):
        if not items:
            errors.append(f"Блок «{block}» пуст.")
    blank = [where for where, text in _text_parts(card) if not text.strip()]
    if blank:
        errors.append(f"Пустые пункты: {', '.join(sorted(set(blank)))}.")
    return errors


def _check_facts(card: CardText, product: ProductInput, analysis: Analysis) -> list[str]:
    allowed = extract_facts(_input_text(product))
    errors, seen = [], set()
    for where, text in _text_parts(card):
        for fragment in unknown_facts(text, allowed):
            if fragment not in seen:
                seen.add(fragment)
                errors.append(f"«{fragment}» ({where}) нет во входных данных — значение, знак или единица не совпадают.")
        for word in MARKETING.findall(text):
            errors.append(f"Рекламная оценка «{word}» ({where}) — уберите непроверяемое утверждение.")
    for field in analysis.absent:
        places = sorted({where for where, text in _text_parts(card) if field.text_pattern.search(text)})
        if places:
            errors.append(
                f"Упомянута характеристика без данных во входе — {field.label} ({', '.join(places)})."
            )
    return errors


def check_card(card: CardText, product: ProductInput, analysis: Analysis) -> tuple[str, ...]:
    return (*_check_shape(card), *_check_facts(card, product, analysis))
