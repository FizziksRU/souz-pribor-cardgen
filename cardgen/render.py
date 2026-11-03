"""Результат в Markdown — в том виде, в каком его копирует контент-менеджер."""

from .schema import CardResult, CardText

STATUS_TEXT = {
    "ok": "готово",
    "needs_review": "требует правки — см. «Замечания проверки»",
    "insufficient_data": "недостаточно данных для карточки",
    "dry_run": "режим без модели (--dry-run), тексты не сформированы",
    "error": "ошибка",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items]


def _section(title: str, body: list[str]) -> list[str]:
    return ["", f"## {title}", "", *body]


def _specs_table(result: CardResult) -> list[str]:
    rows = ["| Характеристика | Значение |", "|---|---|"]
    for row in result.key_specs:
        value = _cell(row.value) if row.has_data else f"**{_cell(row.value)}**"
        rows.append(f"| {_cell(row.name)} | {value} |")
    return rows


def _texts(card: CardText) -> list[str]:
    title, description = card.seo_title.strip(), card.seo_description.strip()
    return [
        *_section(f"SEO-заголовок ({len(title)} симв.)", [title]),
        *_section(f"SEO-описание ({len(description)} симв.)", [description]),
    ]


def _commercial(card: CardText) -> list[str]:
    return [
        *_section("Поисковые запросы", _bullets(card.search_queries)),
        *_section(
            "Для менеджера",
            ["**Кому подходит**", "", *_bullets(card.commercial.audience), "",
             "**Какие задачи решает**", "", *_bullets(card.commercial.tasks)],
        ),
    ]


def to_markdown(result: CardResult) -> str:
    card = result.card
    lines = [f"# {result.product_name}", "", f"Тип прибора: {result.kind}. Статус: {STATUS_TEXT[result.status]}."]
    if result.error:
        lines += ["", f"> {_cell(result.error)}"]
    if card:
        lines += _texts(card)
    lines += _section("Ключевые характеристики", _specs_table(result))
    if card:
        lines += _commercial(card)
    clarify = [*result.needs_clarification, *(card.data_issues if card else [])]
    if clarify:
        lines += _section("Требует уточнения", _bullets(clarify))
    if result.missing_fields:
        lines += _section("Нет данных во входе", _bullets(result.missing_fields))
    if result.validation_errors:
        lines += _section("Замечания проверки", _bullets(result.validation_errors))
    return "\n".join(lines) + "\n"
