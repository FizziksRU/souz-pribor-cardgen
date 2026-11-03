import pytest

from cardgen.analysis import analyze
from cardgen.loader import load_product
from cardgen.schema import CardText
from cardgen.validate import CardFormatError, check_card, parse_card
from tests.conftest import PD100, TM510, card_json, make_card, make_description


@pytest.fixture
def pd100():
    product = load_product(PD100)
    return product, analyze(product)


def errors_for(pd100, **overrides):
    product, analysis = pd100
    return check_card(CardText.model_validate(make_card(**overrides)), product, analysis)


def test_valid_card_has_no_errors(pd100):
    assert errors_for(pd100) == ()


def test_title_longer_than_70_is_an_error(pd100):
    errors = errors_for(pd100, seo_title="Датчик давления " * 6)

    assert any("заголовок" in e for e in errors)


@pytest.mark.parametrize("length", [600, 1300])
def test_description_length_must_be_800_to_1200(pd100, length):
    errors = errors_for(pd100, seo_description=make_description(length))

    assert any("описание" in e.lower() for e in errors)


@pytest.mark.parametrize(
    "queries",
    [["датчик давления", "ПД100"], ["а", "б", "в", "г", "д", "е"], ["датчик ПД100", "Датчик ПД100", "ПД100 цена"]],
)
def test_queries_count_and_uniqueness(pd100, queries):
    errors = errors_for(pd100, search_queries=queries)

    assert any("запрос" in e for e in errors)


def test_number_absent_from_input_is_an_error(pd100):
    errors = errors_for(pd100, seo_title="Датчик ОВЕН ПД100 на 2,5 МПа")

    assert any("2,5" in e for e in errors)


def test_numbers_match_with_comma_or_dot(pd100):
    assert errors_for(pd100, seo_title="Датчик ОВЕН ПД100, ВПИ 1.6 МПа, IP65") == ()


def test_missing_field_must_not_appear_in_text(pd100):
    # 0,5 есть в названии модели, поэтому проверка чисел это не поймает — ловит чек-лист
    errors = errors_for(pd100, seo_title="Датчик ОВЕН ПД100, погрешность 0,5 %")

    assert any("Класс точности" in e for e in errors)


def test_marketing_words_are_errors(pd100):
    errors = errors_for(pd100, commercial={"audience": ["лучший выбор для котельных"], "tasks": ["контроль давления"]})

    assert any("рекламн" in e.lower() for e in errors)


def test_empty_commercial_lists_are_errors(pd100):
    errors = errors_for(pd100, commercial={"audience": [], "tasks": ["контроль давления"]})

    assert any("кому подходит" in e for e in errors)


def test_data_issues_may_mention_missing_fields(pd100):
    assert errors_for(pd100, data_issues=["Погрешность 0,5 % есть только в обозначении — уточнить"]) == ()


@pytest.mark.parametrize(
    "title, fragment",
    [
        ("Датчик ОВЕН ПД100, IP80", "IP80"),  # 80 есть во входе, но как «+80 °С»
        ("Датчик ПД100 для сред до +200 °С", "+200 °С"),  # 200 есть, но как «200 %»
        ("ПД100: 1,6 МПа (16 кгс/см²)", "16 кгс/см²"),  # пересчёт единиц
        ("Датчик ПД100, от +40 °С", "+40 °С"),  # во входе «–40»
    ],
)
def test_number_with_wrong_unit_code_or_sign_is_an_error(pd100, title, fragment):
    errors = errors_for(pd100, seo_title=title)

    assert any(fragment in e for e in errors), errors


def test_range_unit_applies_to_both_ends(pd100):
    assert errors_for(pd100, seo_title="Датчик ОВЕН ПД100, среда от –40 до +100 °С") == ()


def test_missing_medium_temperature_cannot_be_mentioned_loosely():
    product = load_product(PD100).model_copy(update={"specs": {
        k: v for k, v in load_product(PD100).specs.items() if k != "температура измеряемой среды"
    }})
    errors = check_card(
        CardText.model_validate(make_card(seo_title="Датчик ОВЕН ПД100, температура среды –40…+80 °С")),
        product,
        analyze(product),
    )

    assert any("Температура измеряемой среды" in e for e in errors)


def test_high_precision_claim_is_caught_when_accuracy_is_absent(pd100):
    errors = errors_for(pd100, seo_title="Высокоточный датчик ОВЕН ПД100")

    assert any("Класс точности" in e for e in errors)


def test_superlative_with_prefix_is_marketing(pd100):
    errors = errors_for(pd100, seo_title="Наилучший датчик ОВЕН ПД100")

    assert any("рекламн" in e.lower() for e in errors)


def test_blank_items_are_errors(pd100):
    errors = errors_for(pd100, search_queries=["датчик ОВЕН ПД100", "ПД100 купить", "  "])

    assert any("Пустые пункты" in e for e in errors)


def test_parse_card_accepts_code_fence():
    card = parse_card("Вот ответ:\n```json\n" + card_json() + "\n```")

    assert card.seo_title.startswith("Датчик давления")


@pytest.mark.parametrize("raw", ["не JSON", "{\"seo_title\": \"x\"}", "[]"])
def test_parse_card_rejects_bad_answers(raw):
    with pytest.raises(CardFormatError):
        parse_card(raw)


def test_tm510_check_uses_its_own_numbers():
    product = load_product(TM510)
    card = CardText.model_validate(
        make_card(
            seo_title="Манометр РОСМА ТМ-510Р.00 0–2,5 МПа G1/2, Ø100 мм",
            search_queries=["манометр ТМ-510", "манометр РОСМА 100 мм", "манометр G1/2 0–2,5 МПа"],
        )
    )

    assert check_card(card, product, analyze(product)) == ()
