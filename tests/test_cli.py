import json

import pytest

from cardgen import cli
from tests.conftest import PD100, TM510


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_dry_run_writes_json_markdown_and_prompt(tmp_path):
    code = cli.main(["--dry-run", "--out", str(tmp_path), str(PD100), str(TM510)])

    assert code == 0
    stem = PD100.stem
    data = json.loads((tmp_path / f"{stem}.card.json").read_text(encoding="utf-8"))
    assert data["status"] == "dry_run"
    assert "prompt" not in data
    assert (tmp_path / f"{stem}.card.md").read_text(encoding="utf-8").startswith("# ")
    assert "<данные_товара>" in (tmp_path / f"{stem}.prompt.txt").read_text(encoding="utf-8")
    assert (tmp_path / f"{TM510.stem}.card.md").exists()


def test_glob_pattern_is_expanded(tmp_path):
    code = cli.main(["--dry-run", "--out", str(tmp_path), str(PD100.parent / "pd100*.json")])

    assert code == 0
    assert (tmp_path / f"{PD100.stem}.card.md").exists()


def test_unmatched_pattern_is_reported(tmp_path, capsys):
    code = cli.main(["--dry-run", "--out", str(tmp_path), str(tmp_path / "net*.json")])

    assert code == 1
    assert "не найден" in capsys.readouterr().err


def test_broken_file_fails_but_others_are_processed(tmp_path, capsys):
    broken = tmp_path / "broken.json"
    broken.write_text('{"name": "Манометр",}', encoding="utf-8")

    code = cli.main(["--dry-run", "--out", str(tmp_path / "out"), str(broken), str(PD100)])

    assert code == 1
    assert "строка 1" in capsys.readouterr().err
    assert (tmp_path / "out" / f"{PD100.stem}.card.md").exists()


def test_without_model_settings_explains_dry_run(tmp_path, capsys):
    code = cli.main(["--out", str(tmp_path), str(PD100)])

    assert code == 2
    assert "--dry-run" in capsys.readouterr().err


def test_same_names_from_different_folders_are_refused(tmp_path, capsys):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "x.json").write_bytes(PD100.read_bytes())

    code = cli.main(["--dry-run", "--out", str(tmp_path / "out"), str(tmp_path / "a/x.json"), str(tmp_path / "b/x.json")])

    assert code == 1
    assert "перезаписали бы" in capsys.readouterr().err
    assert not (tmp_path / "out" / "x.card.md").exists()


def test_existing_file_with_bracket_is_not_a_pattern(tmp_path):
    odd = tmp_path / "датчик [копия].json"
    odd.write_bytes(PD100.read_bytes())

    assert cli.main(["--dry-run", "--out", str(tmp_path), str(odd)]) == 0


def test_insufficient_data_is_written_not_failed(tmp_path):
    sparse = tmp_path / "sparse.json"
    sparse.write_text('{"name": "Манометр ДМ 02", "specs": {"класс точности": ""}}', encoding="utf-8")

    code = cli.main(["--dry-run", "--out", str(tmp_path), str(sparse)])

    assert code == 0
    assert json.loads((tmp_path / "sparse.card.json").read_text(encoding="utf-8"))["status"] == "insufficient_data"
