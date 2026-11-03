import httpx2
import pytest

from cardgen import compare
from cardgen.polza import PolzaClient
from tests.conftest import PD100, TM510, card_json
from tests.test_polza import KEY, Server, failure, reply


@pytest.fixture
def fake_polza(monkeypatch):
    """Подставной polza.ai: у каждой модели свой сервер и свой план ответов; сеть не нужна."""
    plan, servers = {}, {}

    def from_env(env):
        model = env["LLM_MODEL"]
        server = servers.setdefault(model, Server(*plan.get(model, [])))
        return PolzaClient(KEY, model, http_client=httpx2.Client(transport=httpx2.MockTransport(server)))

    monkeypatch.setattr(compare.PolzaClient, "from_env", staticmethod(from_env))
    monkeypatch.setattr(compare, "load_dotenv", lambda *a, **k: False)
    plan["servers"] = servers
    return plan


def run(tmp_path, *args):
    return compare.main(["--out", str(tmp_path), *args])


def test_each_model_gets_own_folder_and_summary(tmp_path, fake_polza):
    fake_polza.update({
        "a/good": [reply(card_json(), cost=1.25)],
        "b/bad": [failure(402, code="INSUFFICIENT_BALANCE")],
    })

    code = run(tmp_path, "--models", "a/good, b/bad", str(PD100))

    assert code == 1  # у второй модели карточка с ошибкой
    assert (tmp_path / "a_good" / f"{PD100.stem}.card.md").exists()
    assert (tmp_path / "b_bad" / f"{PD100.stem}.card.json").exists()
    summary = (tmp_path / "sravnenie.md").read_text(encoding="utf-8")
    assert f"| `a/good` | {PD100.stem} | ok | 1 | 0 | 1 |" in summary
    assert "| `a/good` | 1 из 1 | 1.25 |" in summary
    assert "| `b/bad` | 0 из 1 | 0.00 |" in summary
    assert "INSUFFICIENT_BALANCE" in summary


def test_repeated_model_runs_once(tmp_path, fake_polza):
    fake_polza["a/good"] = [reply(card_json())]

    assert run(tmp_path, "--models", "a/good,a/good", str(PD100)) == 0
    assert len(fake_polza["servers"]["a/good"].requests) == 1


def test_failed_write_keeps_other_cards_and_summary(tmp_path, fake_polza, monkeypatch):
    fake_polza.update({"a/good": [reply(card_json())], "b/full": [reply(card_json())]})
    real_write = compare.write_result

    def write(result, source, out_dir):
        if out_dir.name == "b_full":
            raise OSError("нет места на диске")
        return real_write(result, source, out_dir)

    monkeypatch.setattr(compare, "write_result", write)

    code = run(tmp_path, "--models", "a/good,b/full", str(PD100))

    assert code == 1
    summary = (tmp_path / "sravnenie.md").read_text(encoding="utf-8")
    assert "| `a/good` | 1 из 1 |" in summary
    assert "карточка не записана: нет места на диске" in summary


def test_crashed_task_is_reported_not_lost(tmp_path, fake_polza, monkeypatch):
    fake_polza["a/good"] = [reply(card_json())]
    real_run = compare._run_one

    def flaky(model, source, product, out):
        if source == TM510:
            raise RuntimeError("неожиданное")
        return real_run(model, source, product, out)

    monkeypatch.setattr(compare, "_run_one", flaky)

    code = run(tmp_path, "--models", "a/good", str(PD100), str(TM510))

    assert code == 1
    summary = (tmp_path / "sravnenie.md").read_text(encoding="utf-8")
    assert f"| `a/good` | {TM510.stem} | сбой |" in summary
    assert "задача упала: RuntimeError('неожиданное')" in summary


def test_interrupted_run_still_writes_summary(tmp_path, fake_polza, monkeypatch):
    fake_polza["a/good"] = [reply(card_json()), reply(card_json())]
    real_wait = compare.wait

    def impatient(pending, timeout):
        real_wait(pending)  # дать начатым задачам доработать, как это делает shutdown
        raise KeyboardInterrupt

    monkeypatch.setattr(compare, "wait", impatient)

    code = run(tmp_path, "--models", "a/good", str(PD100))

    assert code == 1
    summary = (tmp_path / "sravnenie.md").read_text(encoding="utf-8")
    assert "Прогон прерван" in summary and f"| `a/good` | {PD100.stem} | ok |" in summary


def test_dry_run_checks_and_plans_without_requests(tmp_path, fake_polza, capsys):
    code = run(tmp_path, "--dry-run", "--models", "a/x,b/y", str(PD100), str(TM510))

    assert code == 0
    assert f"= 4 карточек, до {4 * compare.CALLS_PER_CARD} платных запросов" in capsys.readouterr().out
    assert all(not server.requests for server in fake_polza["servers"].values())
    assert not (tmp_path / "sravnenie.md").exists()


def test_model_name_cannot_escape_output_folder(tmp_path, fake_polza):
    assert run(tmp_path, "--dry-run", "--models", r"..\..\evil", str(PD100)) == 0
    assert (tmp_path / "_.._evil").is_dir()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--models", "a/b_c,a_b/c"], "попали бы в одну папку"),
        (["--models", " , "], "Не указано ни одной модели"),
    ],
)
def test_refused_before_any_request(tmp_path, fake_polza, capsys, args, message):
    assert run(tmp_path, *args, str(PD100)) == 1
    assert message in capsys.readouterr().err


def test_unmatched_pattern_and_same_names_are_refused(tmp_path, fake_polza, capsys):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "x.json").write_bytes(PD100.read_bytes())

    assert run(tmp_path, "--models", "a/x", str(tmp_path / "net*.json")) == 1
    assert run(tmp_path, "--models", "a/x", str(tmp_path / "a/x.json"), str(tmp_path / "b/x.json")) == 1
    err = capsys.readouterr().err
    assert "Не найдено ни одного файла" in err and "Одинаковые имена файлов" in err


def test_missing_key_stops_before_any_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    code = run(tmp_path, "--models", "a/x", str(PD100))

    assert code == 2
    assert "LLM_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "sravnenie.md").exists()
