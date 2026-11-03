"""Командная строка: python -m cardgen vhod/*.json [--out vyhod] [--dry-run]"""

import argparse
import glob
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .llm import LLMClient, LLMConfigError, client_from_env
from .loader import InputError, load_product
from .pipeline import generate_card
from .render import to_markdown
from .schema import CardResult

EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cardgen",
        description="Черновики карточек товаров КИП по JSON с данными прибора.",
    )
    parser.add_argument("files", nargs="+", help="JSON-файлы с данными товаров (можно шаблон vhod/*.json)")
    parser.add_argument("--out", type=Path, default=Path("vyhod"), help="папка для результатов (по умолчанию vyhod)")
    parser.add_argument("--dry-run", action="store_true", help="не вызывать модель: разбор входа и собранный промпт")
    return parser


def _expand(patterns: list[str]) -> tuple[list[Path], list[str]]:
    """Раскрывает шаблоны сами: PowerShell и cmd не делают этого за программу.

    Возвращает найденные файлы и шаблоны, под которые не подошёл ни один файл.
    """
    paths: list[Path] = []
    unmatched: list[str] = []
    for pattern in patterns:
        if Path(pattern).exists() or not any(ch in pattern for ch in "*?["):
            paths.append(Path(pattern))  # существующий файл с «[» в имени — не шаблон
            continue
        found = sorted(glob.glob(pattern))
        paths.extend(Path(p) for p in found)
        if not found:
            unmatched.append(pattern)
    return paths, unmatched


def _write(result: CardResult, source: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = result.model_dump(mode="json", exclude={"prompt"})
    # newline="\n": одинаковые файлы на Windows и Linux, без CRLF
    (out_dir / f"{source.stem}.card.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    markdown = out_dir / f"{source.stem}.card.md"
    markdown.write_text(to_markdown(result), encoding="utf-8", newline="\n")
    if result.prompt:
        (out_dir / f"{source.stem}.prompt.txt").write_text(result.prompt, encoding="utf-8", newline="\n")
    return markdown


def _process(path: Path, client: LLMClient | None, out_dir: Path) -> bool:
    """Обрабатывает один файл; False — если файл не удалось обработать."""
    try:
        result = generate_card(load_product(path), client)
        markdown = _write(result, path, out_dir)
    except InputError as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return False
    except OSError as exc:
        print(f"{path}: не удалось записать результат: {exc}", file=sys.stderr)
        return False
    print(f"{path.name}: {result.status} -> {markdown}")
    if result.status == "error":
        print(f"  {result.error}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()
    client = None
    if not args.dry_run:
        try:
            client = client_from_env()
        except LLMConfigError as exc:
            print(exc, file=sys.stderr)
            return EXIT_CONFIG
    paths, unmatched = _expand(args.files)
    for pattern in unmatched:
        print(f"{pattern}: не найдено ни одного файла", file=sys.stderr)
    stems = [p.stem for p in paths]
    clashing = [p for p in paths if stems.count(p.stem) > 1]
    for path in clashing:
        print(f"{path}: в запуске несколько файлов «{path.stem}» — их результаты перезаписали бы друг друга", file=sys.stderr)
    results = [_process(path, client, args.out) for path in paths if path not in clashing]
    return EXIT_OK if all(results) and not unmatched and not clashing else EXIT_FAILED
