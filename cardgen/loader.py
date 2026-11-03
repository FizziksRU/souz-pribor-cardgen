"""Чтение и проверка входного JSON с понятными сообщениями об ошибках."""

import json
from pathlib import Path

from pydantic import ValidationError

from .schema import ProductInput


class InputError(ValueError):
    """Вход нельзя обработать: битый JSON, не тот тип или нет обязательных полей."""


class _Repeated(list):
    """Значения ключа, который встретился в одном объекте JSON несколько раз."""


def _keep_repeats(pairs: list[tuple[str, object]]) -> dict:
    # json.loads молча оставляет последнее значение ключа; здесь сохраняются все,
    # чтобы «класс точности» дважды попал в «уточнить», а не потерялся
    merged: dict = {}
    for key, value in pairs:
        if key in merged:
            previous = merged[key]
            merged[key] = _Repeated([*(previous if isinstance(previous, _Repeated) else [previous]), value])
        else:
            merged[key] = value
    return merged


def _describe(error: dict) -> str:
    place = ".".join(str(p) for p in error["loc"]) or "документ"
    kind = error["type"]
    if kind == "missing":
        return f"поле «{place}»: обязательно, но отсутствует"
    if kind == "string_too_short":
        return f"поле «{place}»: не может быть пустым"
    if kind == "dict_type":
        return f"поле «{place}»: ожидается объект вида {{\"характеристика\": \"значение\"}}"
    if kind == "string_type":
        return f"поле «{place}»: ожидается строка"
    message = str(error.get("ctx", {}).get("error") or error["msg"])
    return f"поле «{place}»: {message}" if error["loc"] else message


def product_from_dict(data: object) -> ProductInput:
    if not isinstance(data, dict):
        raise InputError(f"Ожидается JSON-объект с полями товара, получено: {type(data).__name__}")
    repeated = [key for key, value in data.items() if isinstance(value, _Repeated)]
    if repeated:
        raise InputError(f"Поле повторяется: {', '.join(repeated)} — оставьте одно значение.")
    try:
        return ProductInput.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(_describe(e) for e in exc.errors())
        raise InputError(f"Некорректные данные товара: {problems}") from exc


def parse_product(text: str) -> ProductInput:
    try:
        data = json.loads(text, object_pairs_hook=_keep_repeats)
    except json.JSONDecodeError as exc:
        raise InputError(f"Некорректный JSON: {exc.msg} (строка {exc.lineno}, столбец {exc.colno})") from exc
    return product_from_dict(data)


def load_product(path: Path) -> ProductInput:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise InputError(f"Файл не найден: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"Не удалось прочитать {path}: {exc}") from exc
    return parse_product(text)
