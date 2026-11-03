"""Сравнение моделей на одних и тех же товарах.

python -m cardgen.compare --models anthropic/claude-sonnet-5.5,openai/gpt-6-luna-pro vhod/pd100*.json vhod/tm-510*.json

Карточки каждой модели пишутся в <out>/<модель>/, сводная таблица — в <out>/sravnenie.md.
Ключ и настройки (LLM_REASONING, LLM_TEMPERATURE) — из .env, одни на все модели.
--dry-run проверяет настройки, файлы и папки и печатает план без платных запросов.
"""

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .cli import EXIT_CONFIG, EXIT_FAILED, EXIT_OK, clashing_stems, expand_patterns, safe_console, write_result
from .llm import LLMConfigError
from .loader import InputError, load_product
from .pipeline import MAX_ATTEMPTS, RETRY_DELAYS, generate_card
from .polza import PolzaClient, Usage
from .schema import CardResult, ProductInput

log = logging.getLogger(__name__)

PARALLEL = 3  # polza.ai отвечает 402 и на слишком много параллельных запросов
CALLS_PER_CARD = MAX_ATTEMPTS * (1 + len(RETRY_DELAYS))  # худший случай: каждая попытка с повторами


class Refused(Exception):
    """Сравнение нельзя начинать; code — код выхода."""

    def __init__(self, message: str, code: int = EXIT_FAILED):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Run:
    model: str
    source: Path
    result: CardResult | None
    spent: Usage
    seconds: float
    problem: str | None = None  # сбой вне конвейера: карточка не записана или задача упала


def _folder(model: str) -> str:
    """Папка модели: только буквы, цифры, точка и дефис, не с точки — без выхода за пределы --out."""
    return re.sub(r"[^\w.-]", "_", model).lstrip(".") or "_"


def _client(model: str) -> PolzaClient:
    return PolzaClient.from_env({**os.environ, "LLM_MODEL": model})


def _run_one(model: str, source: Path, product: ProductInput, out: Path) -> Run:
    client = _client(model)  # свой клиент на задачу: счётчик расхода не делится между потоками
    started = time.perf_counter()
    result = generate_card(product, client)
    seconds = time.perf_counter() - started
    problem = None
    try:
        write_result(result, source, out / _folder(model))
    except OSError as exc:
        problem = f"карточка не записана: {exc}"
    print(f"{model} | {source.name}: {result.status}, {seconds:.0f} с, {client.usage.cost_rub:.2f} ₽", flush=True)
    return Run(model, source, result, client.usage, seconds, problem)


def _collect(model: str, source: Path, future: Future) -> Run:
    try:
        return future.result()
    except Exception as exc:  # сбой одной задачи не должен терять уже оплаченные карточки других
        log.error("%s | %s: задача упала", model, source.name, exc_info=exc)
        return Run(model, source, None, Usage(), 0.0, f"задача упала: {exc!r}")


def _prepare(args) -> tuple[list[str], list[tuple[Path, ProductInput]]]:
    """Всё, что можно проверить до первого платного запроса."""
    models = list(dict.fromkeys(m.strip() for m in args.models.split(",") if m.strip()))
    if not models:
        raise Refused("Не указано ни одной модели.")
    folders: dict[str, str] = {}
    for model in models:
        other = folders.setdefault(_folder(model), model)
        if other != model:
            raise Refused(f"Модели «{other}» и «{model}» попали бы в одну папку {_folder(model)}.")
    paths, unmatched = expand_patterns(args.files)
    if unmatched:
        raise Refused(f"Не найдено ни одного файла: {', '.join(unmatched)}.")
    if clashing := clashing_stems(paths):
        raise Refused(f"Одинаковые имена файлов — карточки перезаписали бы друг друга: {', '.join(map(str, clashing))}.")
    try:
        products = [(path, load_product(path)) for path in paths]
        for model in models:
            _client(model)
    except InputError as exc:
        raise Refused(str(exc)) from exc
    except LLMConfigError as exc:
        raise Refused(str(exc), EXIT_CONFIG) from exc
    try:
        for folder in folders:
            (args.out / folder).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Refused(f"Не удалось создать папку результатов: {exc}") from exc
    return models, products


def _execute(models: list[str], products: list[tuple[Path, ProductInput]], out: Path) -> tuple[list[Run], bool]:
    """Возвращает завершённые прогоны и признак, что сравнение прервали."""
    tasks = [(model, source, product) for model in models for source, product in products]
    pool = ThreadPoolExecutor(max_workers=min(PARALLEL, len(tasks)))
    futures = [(model, source, pool.submit(_run_one, model, source, product, out)) for model, source, product in tasks]
    interrupted = False
    try:
        pending = {future for _, _, future in futures}
        while pending:  # ожидание с таймаутом: так Ctrl+C доходит и в Windows
            _, pending = wait(pending, timeout=1.0)
    except KeyboardInterrupt:
        interrupted = True
        print("Прервано: новые запросы отменены, начатые доводятся до конца.", file=sys.stderr, flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    runs = [_collect(model, source, future) for model, source, future in futures if not future.cancelled()]
    return runs, interrupted


def _cost(usage: Usage) -> str:
    return f"≥ {usage.cost_rub:.2f}" if usage.unpriced else f"{usage.cost_rub:.2f}"


def _row(run: Run) -> str:
    result, card = run.result, run.result.card if run.result else None
    cells = [
        f"`{run.model}`",
        run.source.stem,
        result.status if result else "сбой",
        str(result.attempts) if result else "—",
        str(len(result.validation_errors)) if result else "—",
        str(run.spent.requests),
        str(len(card.seo_title)) if card else "—",
        str(len(card.seo_description)) if card else "—",
        str(len(card.search_queries)) if card else "—",
        str(run.spent.prompt_tokens + run.spent.completion_tokens),
        _cost(run.spent),
        f"{run.seconds:.0f}",
    ]
    return "| " + " | ".join(cells) + " |"


def _total(model: str, runs: list[Run]) -> str:
    mine = [run for run in runs if run.model == model]
    ok = sum(bool(run.result) and run.result.status == "ok" for run in mine)
    spent = sum((run.spent for run in mine), Usage())
    return f"| `{model}` | {ok} из {len(mine)} | {_cost(spent)} | {sum(run.seconds for run in mine):.0f} |"


def _remarks(run: Run) -> list[str]:
    result = run.result
    found = [*(result.validation_errors if result else []), *([result.error] if result and result.error else [])]
    return [f"* `{run.model}`, {run.source.stem}: {text}" for text in [*found, *([run.problem] if run.problem else [])]]


def _summary(runs: list[Run], interrupted: bool) -> str:
    header = (
        "| Модель | Товар | Статус | Попыток | Замечаний | Вызовов API | Заголовок, симв. | Описание, симв. "
        "| Поисковых фраз | Токенов | ₽ | Секунд |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    remarks = [line for run in runs for line in _remarks(run)]
    parts = [
        "# Сравнение моделей\n",
        *(["**Прогон прерван: в сводке только завершённые карточки.**\n"] if interrupted else []),
        "«≥» — стоимость части запросов неизвестна (нет в ответе или связь оборвалась).\n",
        header,
        *map(_row, runs),
        "\n## Итого по моделям\n",
        "| Модель | Прошли проверку | ₽ | Секунд |\n|---|---|---|---|",
        *(_total(model, runs) for model in dict.fromkeys(run.model for run in runs)),
        "\n## Замечания проверки и ошибки\n",
        *(remarks or ["Нет."]),
    ]
    return "\n".join(parts) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cardgen.compare", description="Сравнение моделей polza.ai.")
    parser.add_argument("files", nargs="+", help="JSON-файлы товаров (можно шаблон vhod/*.json)")
    parser.add_argument("--models", required=True, help="идентификаторы моделей polza.ai через запятую")
    parser.add_argument("--out", type=Path, default=Path("vyhod/sravnenie"), help="папка результатов")
    parser.add_argument("--dry-run", action="store_true", help="только проверка и план, без платных запросов")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    safe_console()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    load_dotenv()
    try:
        models, products = _prepare(args)
    except Refused as exc:
        print(exc, file=sys.stderr)
        return exc.code
    cards = len(models) * len(products)
    print(f"{len(models)} мод. × {len(products)} тов. = {cards} карточек, до {cards * CALLS_PER_CARD} платных запросов.")
    if args.dry_run:
        return EXIT_OK
    runs, interrupted = _execute(models, products, args.out)
    summary = args.out / "sravnenie.md"
    try:
        summary.write_text(_summary(runs, interrupted), encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"Сводка не записана: {exc}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Сводка: {summary}")
    failed = interrupted or any(run.problem or run.result.status == "error" for run in runs)
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
