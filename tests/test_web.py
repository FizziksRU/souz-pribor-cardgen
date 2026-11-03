import pytest
from fastapi.testclient import TestClient

from cardgen import web
from tests.conftest import PD100

JSON = {"Content-Type": "application/json"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    return TestClient(web.app)


def post(client, body: bytes, query: str = "?dry_run=true", headers=JSON):
    return client.post("/api/card" + query, content=body, headers=headers)


def test_index_page_has_form(client):
    page = client.get("/")

    assert page.status_code == 200
    assert "<textarea" in page.text
    assert "dry-run" in page.text


def test_dry_run_returns_result_and_markdown(client):
    reply = post(client, PD100.read_bytes())

    assert reply.status_code == 200
    body = reply.json()
    assert body["result"]["status"] == "dry_run"
    assert body["markdown"].startswith("# Преобразователь избыточного давления")


def test_utf8_bom_is_accepted(client):
    assert post(client, b"\xef\xbb\xbf" + PD100.read_bytes()).status_code == 200


def test_broken_json_is_422_with_reason(client):
    reply = post(client, '{"name": "x",}'.encode())

    assert reply.status_code == 422
    assert "строка 1" in reply.json()["error"]


def test_not_utf8_is_422(client):
    assert post(client, '{"name": "Манометр"}'.encode("cp1251")).status_code == 422


def test_only_json_content_type_is_accepted(client):
    reply = post(client, PD100.read_bytes(), headers={"Content-Type": "text/plain"})

    assert reply.status_code == 415


def test_without_model_settings_is_503(client):
    reply = post(client, PD100.read_bytes(), query="")

    assert reply.status_code == 503
    assert "--dry-run" in reply.json()["error"]


def test_oversized_body_is_413(client):
    assert post(client, b" " * (web.MAX_BODY + 1)).status_code == 413


def test_unexpected_failure_is_json_500(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("сбой")

    monkeypatch.setattr(web, "generate_card", broken)
    reply = post(TestClient(web.app, raise_server_exceptions=False), PD100.read_bytes())

    assert reply.status_code == 500
    assert "Внутренняя ошибка" in reply.json()["error"]
