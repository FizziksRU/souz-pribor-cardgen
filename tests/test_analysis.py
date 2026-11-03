from cardgen.analysis import NO_DATA, analyze
from cardgen.loader import load_product, product_from_dict
from tests.conftest import PD100, TM510


def test_detects_instrument_kind():
    assert analyze(load_product(TM510)).kind == "манометр"
    assert analyze(load_product(PD100)).kind == "датчик давления"
    assert analyze(product_from_dict({"name": "Мультиметр АКИП-2204", "specs": {"IP": "IP54"}})).kind == "прибор"


def test_pd100_has_no_accuracy_and_says_so():
    result = analyze(load_product(PD100))

    assert result.missing == ("Класс точности / погрешность",)
    row = next(r for r in result.key_specs if r.name == "Класс точности / погрешность")
    assert row.value == NO_DATA
    assert row.has_data is False


def test_key_specs_keep_input_order_and_values():
    result = analyze(load_product(PD100))

    assert result.key_specs[0].name == "Производитель"
    assert result.key_specs[0].value == "ОВЕН"
    assert result.key_specs[1].name == "Тип"
    assert [r.value for r in result.key_specs if r.name == "Верхний предел измерения"] == ["1,6 МПа"]


def test_tm510_is_complete_but_accuracy_needs_clarification():
    result = analyze(load_product(TM510))

    assert result.missing == ()
    assert any("Класс точности" in note and "1,5; 1,0" in note for note in result.clarify)


def test_missing_manufacturer_and_description_are_reported():
    product = product_from_dict({"name": "Манометр ДМ 02", "specs": {"диапазон измерений": "0–1,6 МПа"}})

    result = analyze(product)

    assert "Производитель" in result.missing
    assert "Описание" in result.missing
    assert result.key_specs[0].value == NO_DATA


def test_empty_input_fields_are_listed_for_clarification():
    product = product_from_dict({"name": "Манометр", "specs": {"класс точности": "", "диаметр корпуса": "100 мм"}})

    result = analyze(product)

    assert any("класс точности" in note for note in result.clarify)


def test_enough_data_needs_three_specs_or_real_description():
    two = {"диапазон измерений": "0–1,6 МПа", "класс точности": "1,5"}
    three = {**two, "диаметр корпуса": "100 мм"}

    assert analyze(product_from_dict({"name": "Манометр"})).enough_data is False
    assert analyze(product_from_dict({"name": "Манометр", "description": "Показывающий."})).enough_data is False
    assert analyze(product_from_dict({"name": "Манометр", "specs": two})).enough_data is False
    assert analyze(product_from_dict({"name": "Манометр", "specs": three})).enough_data is True
    assert analyze(product_from_dict({"name": "Манометр", "description": "Показывающий манометр. " * 8})).enough_data


def test_fact_given_only_in_description_is_not_reported_missing():
    product = product_from_dict({
        "name": "Манометр ДМ 02",
        "manufacturer": "Манотомь",
        "specs": {"диапазон измерений": "0–1,6 МПа"},
        "description": "Диаметр корпуса 63 мм. Корпус IP40.",
    })

    result = analyze(product)

    assert "Диаметр корпуса" not in result.missing
    assert "Степень защиты (IP)" not in result.missing
    assert any("Диаметр корпуса: есть только в описании" in note for note in result.clarify)


def test_accuracy_claim_without_value_in_description_stays_missing():
    product = load_product(PD100).model_copy(update={"description": "Исключительная точность во всём диапазоне."})

    assert "Класс точности / погрешность" in analyze(product).missing


def test_detects_manometer_family_and_long_transmitter_names():
    def kind(name):
        return analyze(product_from_dict({"name": name})).kind

    assert kind("Мановакуумметр ДА2005") == "манометр"
    assert kind("Вакуумметр ВП3-У") == "манометр"
    assert kind("Тягонапоромер ТНМП-52") == "манометр"
    assert kind("Датчик избыточного и абсолютного давления") == "датчик давления"
