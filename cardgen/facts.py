"""Факты из текста для сверки с входом: числа со знаком, числа с единицами, коды (IP, резьбы).

Сверяется не «мешок чисел»: «IP80» не проходит из-за «+80 °С», «+200 °С» — из-за «200 %»,
«+40» — из-за «–40». Единица после диапазона относится к обоим концам: «–40…+100 °С».
"""

import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

# Порядок важен: длинные единицы раньше коротких («кгс/см²» раньше «кг», «Вт» раньше «В»)
UNITS = (
    ("кгс/см2", r"кгс/см[²2]"),
    ("мпа", r"мпа|mpa"),
    ("кпа", r"кпа|kpa"),
    ("па", r"па|pa"),
    ("бар", r"бар|bar"),
    ("мка", r"мк[аa]|µa"),
    ("ма", r"[мm][аa]"),
    ("вт", r"вт|w"),
    ("в", r"(?-i:В|V)"),  # строчная «в» — предлог, а не вольты
    ("ком", r"ком|kω"),
    ("ом", r"ом|ω"),
    ("°с", r"°\s?[сc]"),
    ("мм", r"мм|mm"),
    ("кг", r"кг|kg"),
    ("гц", r"гц|hz"),
    ("%", r"%"),
    ("ч", r"ч"),
    ("мес", r"мес"),
    ("год", r"лет|год[аы]?"),
)
UNIT_RES = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in UNITS)

QUANTITY = re.compile(
    r"(?:(?<!\w)(?P<sign>[-–−+]))?(?P<num>\d+(?:[.,]\d+)?)"
    r"(?:\s?(?P<unit>" + "|".join(p for _, p in UNITS) + r")(?![а-яёa-z]))?",
    re.IGNORECASE,
)
RANGE_GAP = re.compile(r"\s*(?:…|\.{2,3}|–|—|-|−|÷|/|до)\s*", re.IGNORECASE)
CODE = re.compile(r"\bIP\s?\d{2}\b|\bG\s?\d+(?:/\d+)?\b|[MМ]\d+\s?[xх×]\s?\d+(?:[.,]\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class Quantity:
    sign: str
    value: str
    unit: str | None
    start: int
    end: int


@dataclass(frozen=True)
class Facts:
    numbers: frozenset[tuple[str, str]]
    quantities: frozenset[tuple[str, str, str]]
    codes: frozenset[str]


def _number(token: str) -> str:
    try:
        return format(Decimal(token.replace(",", ".")).normalize(), "f")
    except InvalidOperation:
        return token


def _unit(raw: str | None) -> str | None:
    if not raw:
        return None
    return next(name for name, pattern in UNIT_RES if pattern.fullmatch(raw))


def _code(raw: str) -> str:
    return re.sub(r"\s", "", raw).upper().replace("М", "M").replace("Х", "X").replace("×", "X").replace(",", ".")


def quantities(text: str) -> list[Quantity]:
    found = [
        Quantity(
            sign="-" if m.group("sign") in ("-", "–", "−") else "",
            value=_number(m.group("num")),
            unit=_unit(m.group("unit")),
            start=m.start(),
            end=m.end(),
        )
        for m in QUANTITY.finditer(text)
    ]
    # справа налево, чтобы единица дошла по цепочке «0…1 / 1,6 / 2,5 МПа»
    for i in range(len(found) - 2, -1, -1):
        left, right = found[i], found[i + 1]
        if left.unit is None and right.unit and RANGE_GAP.fullmatch(text[left.end : right.start]):
            found[i] = replace(left, unit=right.unit)
    return found


def extract_facts(text: str) -> Facts:
    found = quantities(text)
    return Facts(
        numbers=frozenset((q.sign, q.value) for q in found),
        quantities=frozenset((q.sign, q.value, q.unit) for q in found if q.unit),
        codes=frozenset(_code(m.group()) for m in CODE.finditer(text)),
    )


def unknown_facts(text: str, allowed: Facts) -> list[str]:
    """Фрагменты текста с числами, единицами или кодами, которых нет во входе."""
    problems = []
    for q in quantities(text):
        known = (q.sign, q.value) in allowed.numbers
        if not known or (q.unit and (q.sign, q.value, q.unit) not in allowed.quantities):
            problems.append(text[q.start : q.end].strip())
    problems.extend(m.group() for m in CODE.finditer(text) if _code(m.group()) not in allowed.codes)
    return problems
