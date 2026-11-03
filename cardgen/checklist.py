"""Чек-листы обязательных характеристик по типам приборов.

key_pattern ищет характеристику среди ключей входа, text_pattern — упоминание
в сгенерированном тексте (чтобы модель не писала о том, чего нет во входе).
"""

import re
from dataclasses import dataclass
from typing import Mapping

from .schema import ProductInput


@dataclass(frozen=True)
class RequiredField:
    label: str
    key_pattern: re.Pattern[str]
    text_pattern: re.Pattern[str]

    def matching_keys(self, specs: Mapping[str, str]) -> list[str]:
        return [key for key in specs if self.key_pattern.search(key)]


@dataclass(frozen=True)
class Checklist:
    kind: str
    fields: tuple[RequiredField, ...]


def _field(label: str, keys: str, text: str | None = None) -> RequiredField:
    return RequiredField(label, re.compile(keys, re.IGNORECASE), re.compile(text or keys, re.IGNORECASE))


RANGE = _field(
    "Диапазон измерений",
    r"диапазон\w*\s+(измерен|показан|давлен)|предел\w*\s+измерен|\bвпи\b",
    r"диапазон\w*\s+(измерен|показан)|предел\w*\s+измерен|\bвпи\b",
)
ACCURACY = _field(
    "Класс точности / погрешность",
    r"класс\w*\s+точност|погрешност|точност",
    # «точный», «высокоточный» — да; «уточнить», «избыточное» — нет
    r"класс\w*\s+точност|погрешност|(?<![а-яё])точн|высокоточн",
)
DIAMETER = _field("Диаметр корпуса", r"диаметр", r"диаметр|[ø⌀]")
CONNECTION = _field(
    "Присоединение к процессу",
    r"резьб|присоедин|подключени\w*\s+к\s+процесс",
    r"резьб|присоедин|штуцер",
)
FITTING = _field(
    "Расположение штуцера",
    r"расположени\w*\s+штуцер",
    r"радиальн|осев\w*\s+(присоедин|штуцер)|расположени\w*\s+штуцер",
)
PROTECTION = _field(
    "Степень защиты (IP)",
    r"степен\w*\s+(пылевлаго|влаго|пыле)?защит|\bip(\s?\d{2})?\b",
    r"степен\w*\s+(пылевлаго|влаго|пыле)?защит|\bip\s?\d",
)
MEDIUM_TEMP = _field(
    "Температура измеряемой среды",
    r"температур\w*\s+(измеряем|рабоч)\w*\s+сред",
    # «температура среды», «рабочей среды», но не «окружающей среды»
    r"температур\w*\s+(?!окружающ)(\w+\s+)?сред|(измеряем|рабоч)\w*\s+сред",
)
AMBIENT_TEMP = _field("Температура окружающей среды", r"температур\w*\s+окружающ")
OUTPUT = _field(
    "Выходной сигнал",
    r"выходн\w*\s+сигнал",
    r"выходн\w*\s+сигнал|(?<![а-яёa-z])[мm][аa](?![а-яёa-z])",  # «мА», «mA», «20мА»
)
POWER = _field("Напряжение питания", r"питани")
ELECTRIC = _field(
    "Электрическое подключение",
    r"соединител|разъ[её]м|кабельн\w*\s+ввод|электрическ\w*\s+подключ",
)

MANOMETER = Checklist("манометр", (RANGE, ACCURACY, DIAMETER, CONNECTION, FITTING, PROTECTION, MEDIUM_TEMP))
TRANSMITTER = Checklist(
    "датчик давления",
    (RANGE, ACCURACY, OUTPUT, POWER, CONNECTION, ELECTRIC, PROTECTION, MEDIUM_TEMP),
)
GENERIC = Checklist("прибор", (RANGE, ACCURACY, AMBIENT_TEMP))

KINDS = (
    (re.compile(r"маномет|вакуумм?етр|напоромер|тягомер", re.IGNORECASE), MANOMETER),
    (re.compile(r"(датчик|преобразовател)\w*\W+(\w+\W+){0,3}давлени", re.IGNORECASE), TRANSMITTER),
)


def detect_checklist(product: ProductInput) -> Checklist:
    """Тип прибора определяется по категории и названию; описание не учитывается,
    потому что в нём часто упоминаются соседние приборы."""
    text = " ".join(filter(None, (product.category, product.name)))
    return next((checklist for pattern, checklist in KINDS if pattern.search(text)), GENERIC)
