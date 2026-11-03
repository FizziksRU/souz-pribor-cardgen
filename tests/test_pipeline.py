from cardgen.llm import LLMError
from cardgen.loader import load_product, product_from_dict
from cardgen.pipeline import generate_card
from tests.conftest import PD100, card_json


def test_dry_run_builds_prompt_without_calling_model():
    result = generate_card(load_product(PD100), client=None)

    assert result.status == "dry_run"
    assert result.card is None
    assert result.attempts == 0
    assert "ПД100-ДИ1,6-111-0,5" in result.prompt
    assert result.missing_fields == ["Класс точности / погрешность"]


def test_good_first_answer_is_ok(fake_llm):
    llm = fake_llm(card_json())

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "ok"
    assert result.attempts == 1
    assert result.card.seo_title.startswith("Датчик давления")
    assert result.validation_errors == []


def test_bad_answer_is_retried_with_errors(fake_llm):
    llm = fake_llm(card_json(seo_title="Датчик " * 15), card_json())

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "ok"
    assert result.attempts == 2
    _, retry_messages = llm.calls[1]
    assert [m.role for m in retry_messages] == ["user", "assistant", "user"]
    assert "заголовок" in retry_messages[-1].content


def test_still_bad_after_retry_needs_review(fake_llm):
    bad = card_json(seo_title="Датчик на 2,5 МПа")
    llm = fake_llm(bad, bad)

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "needs_review"
    assert result.card.seo_title == "Датчик на 2,5 МПа"
    assert any("2,5" in e for e in result.validation_errors)


def test_unparsable_then_good_is_ok(fake_llm):
    llm = fake_llm("Извините, не могу", card_json())

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "ok"
    assert "JSON" in llm.calls[1][1][-1].content


def test_never_parsable_is_error_without_card(fake_llm):
    llm = fake_llm("нет", "всё ещё нет")

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "error"
    assert result.card is None
    assert "JSON" in result.error


def test_llm_failure_is_reported(fake_llm, llm_error):
    llm = fake_llm(llm_error)

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "error"
    assert result.error == "Сервис LLM недоступен"
    assert result.key_specs  # характеристики всё равно собраны


def test_llm_failure_on_retry_keeps_first_card(fake_llm, llm_error):
    llm = fake_llm(card_json(seo_title="Датчик " * 15), llm_error)

    result = generate_card(load_product(PD100), client=llm)

    assert result.status == "needs_review"
    assert result.card is not None
    assert "Сервис LLM недоступен" in result.error


def test_temporary_error_is_retried_after_pause(fake_llm):
    pauses = []
    llm = fake_llm(LLMError("429 Too Many Requests", retryable=True), card_json())

    result = generate_card(load_product(PD100), client=llm, sleep=pauses.append)

    assert result.status == "ok"
    assert pauses == [2.0]


def test_temporary_error_gives_up_after_all_pauses(fake_llm):
    busy = LLMError("503", retryable=True)

    result = generate_card(load_product(PD100), client=fake_llm(busy, busy, busy), sleep=lambda _: None)

    assert result.status == "error"
    assert result.error == "503"


def test_unexpected_client_crash_becomes_error_status(fake_llm):
    result = generate_card(load_product(PD100), client=fake_llm(TimeoutError("read timed out")))

    assert result.status == "error"
    assert "TimeoutError" in result.error


def test_client_returning_none_is_an_error(fake_llm):
    result = generate_card(load_product(PD100), client=fake_llm(None))

    assert result.status == "error"
    assert "NoneType" in result.error


def test_needs_review_keeps_the_card_with_fewer_errors(fake_llm):
    one_error = card_json(seo_title="Датчик на 2,5 МПа")
    many_errors = card_json(seo_title="Датчик " * 15, search_queries=["ПД100"])

    result = generate_card(load_product(PD100), client=fake_llm(one_error, many_errors))

    assert result.status == "needs_review"
    assert result.card.seo_title == "Датчик на 2,5 МПа"


def test_insufficient_data_skips_model(fake_llm):
    llm = fake_llm()

    result = generate_card(product_from_dict({"name": "Манометр ДМ 02", "manufacturer": "Манотомь"}), client=llm)

    assert result.status == "insufficient_data"
    assert llm.calls == []
    assert "Описание" in result.missing_fields
