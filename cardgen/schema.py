"""Схемы данных: вход, тексты от модели и итоговая карточка."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Порядок проверки совпадает с порядком полей во входе — так же идёт и отчёт о пустых полях
FIELD_ORDER = ("manufacturer", "category", "specs", "description")


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _clean_specs(specs: object) -> tuple[object, list[str]]:
    """Возвращает характеристики без пустых значений и список пустых ключей.

    Ключи, совпавшие после чистки пробелов («класс точности» и «класс точности »),
    не затирают друг друга, а сливаются в «1,5; 1,0» — такое значение уйдёт в «уточнить».
    Не-словарь возвращается как есть: его отвергнет проверка типа поля.
    """
    if not isinstance(specs, dict):
        return specs, []
    merged: dict[str, str] = {}
    blank: list[str] = []
    for raw_key, raw_value in specs.items():
        key = _one_line(str(raw_key)) or "(без названия)"
        text = _spec_text(key, raw_value)
        if text is None or key == "(без названия)":
            blank.append(key)
        else:
            merged[key] = f"{merged[key]}; {text}" if key in merged else text
    return merged, blank


def _spec_text(key: str, value: object) -> str | None:
    """Приводит значение характеристики к строке; None — значение пустое."""
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, dict):
        raise ValueError(f"характеристика «{key}»: ожидается текст или число, получено {type(value).__name__}")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_spec_text(key, item) for item in value]
        return "; ".join(p for p in parts if p) or None
    if isinstance(value, str):
        return value.strip() or None
    raise ValueError(f"характеристика «{key}»: неподдерживаемый тип {type(value).__name__}")


class ProductInput(BaseModel):
    """Данные о приборе. Пустые значения выкидываются и запоминаются в empty_fields."""

    model_config = ConfigDict(extra="allow", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1)
    manufacturer: str | None = None
    category: str | None = None
    specs: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    empty_fields: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _drop_empty(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        empty: list[str] = []
        for field in FIELD_ORDER:
            if field not in cleaned:
                continue
            if field == "specs" and cleaned["specs"] is not None:
                cleaned["specs"], blank_specs = _clean_specs(cleaned["specs"])
                empty.extend(f"specs.{key}" for key in blank_specs)
            elif field == "specs":
                cleaned["specs"] = {}
                empty.append("specs")
            elif _is_blank(cleaned[field]):
                cleaned[field] = None
                empty.append(field)
        cleaned["empty_fields"] = tuple(empty)
        return cleaned

    @field_validator("name", mode="before")
    @classmethod
    def _name_is_text(cls, value: object) -> object:
        if value is None:
            raise ValueError("название обязательно")
        return value

    @field_validator("name", "manufacturer", "category")
    @classmethod
    def _single_line(cls, value: str | None) -> str | None:
        # перевод строки в названии сломал бы заголовок карточки и таблицу
        return _one_line(value) if value is not None else None


class Commercial(BaseModel):
    audience: list[str]
    tasks: list[str]


class CardText(BaseModel):
    """Тексты, которые пишет модель. Характеристики сюда не входят — их собирает код."""

    seo_title: str
    seo_description: str
    search_queries: list[str]
    commercial: Commercial
    data_issues: list[str] = Field(default_factory=list)


class SpecRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    value: str
    has_data: bool


Status = Literal["ok", "needs_review", "insufficient_data", "dry_run", "error"]


class CardResult(BaseModel):
    product_name: str
    kind: str
    status: Status
    card: CardText | None = None
    key_specs: list[SpecRow]
    missing_fields: list[str]
    needs_clarification: list[str]
    validation_errors: list[str] = Field(default_factory=list)
    attempts: int = 0
    error: str | None = None
    prompt: str | None = None
