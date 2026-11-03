"""Разбор входа без модели: ключевые характеристики, чего не хватает, что уточнить."""

import re
from dataclasses import dataclass

from .checklist import Checklist, RequiredField, detect_checklist
from .schema import ProductInput, SpecRow

NO_DATA = "данных нет"
ONLY_IN_DESCRIPTION = "есть только в описании"
MULTI_VALUE = re.compile(r";|\s/\s|\sили\s")
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
FIELD_NAMES = {
    "manufacturer": "производитель",
    "category": "категория",
    "specs": "характеристики",
    "description": "описание",
}
# Меньше этого на 800+ символов описания без домыслов не набрать — модель не вызывается
MIN_SPECS = 3
MIN_DESCRIPTION = 150


@dataclass(frozen=True)
class Analysis:
    kind: str
    checklist: Checklist
    absent: tuple[RequiredField, ...]
    described: tuple[RequiredField, ...]
    key_specs: tuple[SpecRow, ...]
    missing: tuple[str, ...]
    clarify: tuple[str, ...]
    enough_data: bool


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _in_description(field: RequiredField, description: str | None) -> bool:
    """Характеристика названа в описании вместе с числом — значит, значение там есть.

    Ищем по строгому key_pattern: широкий text_pattern принял бы «давление рабочей среды … 4…20 мА»
    за температуру среды.
    """
    sentences = SENTENCE_END.split(description or "")
    return any(field.key_pattern.search(s) and re.search(r"\d", s) for s in sentences)


def _key_specs(product: ProductInput, absent: tuple[RequiredField, ...], described: tuple[RequiredField, ...]):
    maker = SpecRow(name="Производитель", value=product.manufacturer or NO_DATA, has_data=bool(product.manufacturer))
    given = [SpecRow(name=_capitalize(k), value=v, has_data=True) for k, v in product.specs.items()]
    in_text = [SpecRow(name=f.label, value=ONLY_IN_DESCRIPTION, has_data=False) for f in described]
    lacking = [SpecRow(name=f.label, value=NO_DATA, has_data=False) for f in absent]
    return (maker, *given, *in_text, *lacking)


def _missing(product: ProductInput, absent: tuple[RequiredField, ...]) -> tuple[str, ...]:
    head = () if product.manufacturer else ("Производитель",)
    tail = () if product.description else ("Описание",)
    return (*head, *(f.label for f in absent), *tail)


def _clarify(product: ProductInput, described: tuple[RequiredField, ...]) -> tuple[str, ...]:
    several = [
        f"{_capitalize(k)}: несколько значений «{v}» — какое относится к этому исполнению?"
        for k, v in product.specs.items()
        if MULTI_VALUE.search(v)
    ]
    in_text = [f"{f.label}: {ONLY_IN_DESCRIPTION} — перенести в характеристики." for f in described]
    blank = [
        f"Поле «{field.removeprefix('specs.') if field.startswith('specs.') else FIELD_NAMES.get(field, field)}» "
        "передано пустым — заполнить или убрать."
        for field in product.empty_fields
    ]
    return (*several, *in_text, *blank)


def analyze(product: ProductInput) -> Analysis:
    checklist = detect_checklist(product)
    not_in_specs = [f for f in checklist.fields if not f.matching_keys(product.specs)]
    described = tuple(f for f in not_in_specs if _in_description(f, product.description))
    absent = tuple(f for f in not_in_specs if f not in described)
    return Analysis(
        kind=checklist.kind,
        checklist=checklist,
        absent=absent,
        described=described,
        key_specs=_key_specs(product, absent, described),
        missing=_missing(product, absent),
        clarify=_clarify(product, described),
        enough_data=len(product.specs) >= MIN_SPECS or len(product.description or "") >= MIN_DESCRIPTION,
    )
