import json

import httpx2
import pytest

from cardgen import cli, polza
from cardgen.llm import LLMConfigError, LLMError, Message, client_from_env
from cardgen.loader import load_product
from cardgen.pipeline import generate_card
from cardgen.polza import PolzaClient
from tests.conftest import PD100, card_json

KEY = "sk-test-secret-key"
MODEL = "anthropic/claude-sonnet-5.5"


def reply(content="{}", finish="stop", cost=1.5):
    return httpx2.Response(200, json={
        "id": "gen_1", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 800, "total_tokens": 2000, "cost_rub": cost},
    })


def failure(status, code="SOME_CODE", message="подробности от сервиса"):
    return httpx2.Response(status, json={"error": {"code": code, "message": message}})


class Server:
    """Подставной polza.ai: отдаёт ответы по очереди и запоминает запросы. Сеть не нужна."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        return item(request) if callable(item) else item

    @property
    def body(self):
        return json.loads(self.requests[-1].content)


def make_client(server, **options):
    return PolzaClient(KEY, MODEL, http_client=httpx2.Client(transport=httpx2.MockTransport(server)), **options)


def ask(client):
    return client.complete("системный промпт", [Message("user", "данные товара")])


def test_request_goes_to_polza_with_key_model_and_json_mode():
    server = Server(reply('{"seo_title": "x"}'))

    answer = ask(make_client(server))

    assert answer == '{"seo_title": "x"}'
    request = server.requests[0]
    assert str(request.url) == "https://polza.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert server.body["model"] == MODEL
    assert server.body["messages"] == [
        {"role": "system", "content": "системный промпт"},
        {"role": "user", "content": "данные товара"},
    ]
    assert server.body["response_format"] == {"type": "json_object"}
    assert server.body["max_tokens"] == polza.MAX_TOKENS
    assert "temperature" not in server.body and "reasoning" not in server.body


def test_reasoning_and_temperature_are_sent_when_set():
    server = Server(reply())

    ask(make_client(server, reasoning="low", temperature=0.3))

    assert server.body["reasoning"] == {"effort": "low"}
    assert server.body["temperature"] == 0.3


def test_usage_is_summed_over_requests():
    client = make_client(Server(reply(cost=1.5), reply(cost=0.25)))

    ask(client)
    ask(client)

    assert client.usage == polza.Usage(requests=2, prompt_tokens=2400, completion_tokens=1600, cost_rub=1.75)


@pytest.mark.parametrize("cost", ["n/a", None, -1, "inf"])
def test_unknown_cost_is_counted_not_hidden_as_zero(cost):
    client = make_client(Server(reply("{}", cost=cost)))

    assert ask(client) == "{}"
    assert client.usage == polza.Usage(requests=1, prompt_tokens=1200, completion_tokens=800, unpriced=1)


def test_cost_with_comma_is_read():
    client = make_client(Server(reply("{}", cost="1,5")))

    ask(client)

    assert client.usage.cost_rub == 1.5 and client.usage.unpriced == 0


def test_openai_environment_headers_do_not_leak_to_polza(monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "org-secret")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-secret")
    server = Server(reply())

    ask(make_client(server))

    headers = server.requests[0].headers
    assert "openai-organization" not in headers and "openai-project" not in headers


def test_custom_openai_headers_are_refused(monkeypatch):
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer sk-openai-real")

    with pytest.raises(LLMConfigError, match="OPENAI_CUSTOM_HEADERS"):
        make_client(Server())


@pytest.mark.parametrize(
    ("status", "retryable", "hint"),
    [
        (400, False, "Ошибка polza.ai"),
        (401, False, "LLM_API_KEY"),
        (402, False, "балансе"),
        (403, False, "запрещён"),
        (404, False, MODEL),
        (408, True, "Ошибка polza.ai"),
        (429, True, "Ошибка polza.ai"),
        (502, True, "Ошибка polza.ai"),
        (503, True, "Ошибка polza.ai"),
        (520, True, "Ошибка polza.ai"),
    ],
)
def test_http_errors_are_explained(status, retryable, hint):
    client = make_client(Server(failure(status, code="SOME_CODE", message="подробности от сервиса")))

    with pytest.raises(LLMError) as caught:
        ask(client)

    text = str(caught.value)
    assert caught.value.retryable is retryable
    assert hint in text and f"HTTP {status}" in text
    assert "подробности от сервиса [SOME_CODE]" in text


def test_key_quoted_by_service_is_masked():
    client = make_client(Server(failure(401, code="UNAUTHORIZED", message=f"Invalid key {KEY}")))

    with pytest.raises(LLMError) as caught:
        ask(client)

    assert KEY not in str(caught.value) and "Invalid key ***" in str(caught.value)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx2.Response(500, text="Internal Server Error"), "Internal Server Error"),
        (httpx2.Response(400, json={"error": "модель не поддерживает response_format"}), "не поддерживает response_format"),
        (httpx2.Response(502, text="<html>" + "x" * 5000 + "</html>"), "<html>"),
    ],
)
def test_error_body_without_standard_shape_is_kept_short(response, expected):
    with pytest.raises(LLMError) as caught:
        ask(make_client(Server(response)))

    text = str(caught.value)
    assert f"HTTP {response.status_code}" in text and expected in text
    assert len(text) < 500


def test_connection_not_established_is_retryable_and_free():
    def offline(request):
        raise httpx2.ConnectError("нет сети", request=request)

    client = make_client(Server(offline))
    with pytest.raises(LLMError, match="Нет связи с polza.ai") as caught:
        ask(client)

    assert caught.value.retryable
    assert client.usage == polza.Usage()


@pytest.mark.parametrize(
    ("broken", "message"),
    [(httpx2.ReadTimeout, "не ответил за 620 с"), (httpx2.ReadError, "оборвалась"), (httpx2.RemoteProtocolError, "оборвалась")],
)
def test_break_after_sending_is_not_retried(broken, message):
    """polza.ai списывает уже сгенерированную часть: повтор заплатил бы второй раз."""

    def cut(request):
        raise broken("обрыв", request=request)

    client = make_client(Server(cut))
    with pytest.raises(LLMError, match=message) as caught:
        ask(client)

    assert not caught.value.retryable
    assert client.usage == polza.Usage(requests=1, unpriced=1)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx2.Response(200, text="<html>Технические работы</html>"), "не в формате API"),
        (
            httpx2.Response(200, json={"id": "gen_1", "object": "chat.completion", "created": 1, "model": MODEL,
                                       "choices": [{"index": 0, "finish_reason": "stop", "message": None}]}),
            "пустой ответ",
        ),
        (
            httpx2.Response(200, json={"id": "gen_1", "object": "chat.completion", "created": 1, "model": MODEL,
                                       "choices": [], "error": {"code": "BAD_GATEWAY", "message": "провайдер упал"}}),
            "провайдер упал [BAD_GATEWAY]",
        ),
    ],
)
def test_odd_success_responses_become_retryable_errors(response, message):
    with pytest.raises(LLMError, match=message.replace("[", r"\[").replace("]", r"\]")) as caught:
        ask(make_client(Server(response)))

    assert caught.value.retryable


def test_truncated_answer_is_not_retried():
    with pytest.raises(LLMError, match="обрезан") as caught:
        ask(make_client(Server(reply('{"seo_title": "Дат', finish="length"))))

    assert not caught.value.retryable


def test_filtered_answer_is_not_retried():
    with pytest.raises(LLMError, match="фильтром") as caught:
        ask(make_client(Server(reply("", finish="content_filter"))))

    assert not caught.value.retryable


@pytest.mark.parametrize(("content", "finish"), [("", "stop"), ("   ", "stop"), (None, "stop"), ("{}", "error")])
def test_empty_or_broken_answer_is_retryable(content, finish):
    with pytest.raises(LLMError, match="пустой ответ") as caught:
        ask(make_client(Server(reply(content, finish=finish))))

    assert caught.value.retryable


def test_answer_without_choices_and_usage_is_retryable():
    bare = httpx2.Response(200, json={"id": "gen_1", "object": "chat.completion", "created": 1, "model": MODEL, "choices": []})
    client = make_client(Server(bare))

    with pytest.raises(LLMError, match="без вариантов") as caught:
        ask(client)

    assert caught.value.retryable
    assert client.usage == polza.Usage(requests=1, unpriced=1)


def test_pipeline_retries_temporary_error_and_builds_card():
    server = Server(failure(503, code="SERVICE_UNAVAILABLE"), reply(card_json()))
    pauses = []

    result = generate_card(load_product(PD100), make_client(server), sleep=pauses.append)

    assert result.status == "ok"
    assert len(server.requests) == 2 and pauses == [2.0]


def test_cli_prints_what_the_run_cost(tmp_path, capsys, monkeypatch):
    client = make_client(Server(reply(card_json(), cost=1.25)))
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(cli, "client_from_env", lambda: client)

    code = cli.main(["--out", str(tmp_path), str(PD100)])

    out = capsys.readouterr().out
    assert code == 0
    assert "ok ->" in out and "(1 запр., 2000 токенов, 1.25 ₽)" in out
    assert "Итого (1 запр., 2000 токенов, 1.25 ₽)" in out


def test_cli_says_when_cost_is_unknown(tmp_path, capsys, monkeypatch):
    client = make_client(Server(reply(card_json(), cost=None)))
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(cli, "client_from_env", lambda: client)

    cli.main(["--out", str(tmp_path), str(PD100)])

    assert "0.00 ₽, у 1 стоимость неизвестна)" in capsys.readouterr().out


def test_settings_come_from_env():
    env = {"LLM_API_KEY": f" {KEY} ", "LLM_MODEL": MODEL, "LLM_REASONING": "Low", "LLM_TEMPERATURE": "0,3"}

    client = PolzaClient.from_env(env)

    assert (client.model, client.reasoning, client.temperature) == (MODEL, "low", 0.3)


def test_missing_key_and_model_are_named():
    with pytest.raises(LLMConfigError, match="LLM_API_KEY, LLM_MODEL"):
        PolzaClient.from_env({"LLM_API_KEY": " ", "LLM_MODEL": ""})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LLM_REASONING", "очень", "LLM_REASONING"),
        ("LLM_TEMPERATURE", "горячо", "получено «горячо»"),
        ("LLM_TEMPERATURE", "3", "получено 3.0"),
    ],
)
def test_bad_optional_settings_are_refused(name, value, message):
    env = {"LLM_API_KEY": KEY, "LLM_MODEL": MODEL, name: value}

    with pytest.raises(LLMConfigError, match=message):
        PolzaClient.from_env(env)


def test_provider_polza_is_registered(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Polza ")
    monkeypatch.setenv("LLM_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    for name in ("LLM_REASONING", "LLM_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)

    assert isinstance(client_from_env(), PolzaClient)


def test_unknown_provider_lists_known_ones(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "other")

    with pytest.raises(LLMConfigError, match="Доступные: polza"):
        client_from_env()
