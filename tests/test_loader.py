import json

import pytest

from cardgen.loader import InputError, load_product, parse_product
from tests.conftest import PD100, TM510


def test_loads_both_sample_products():
    pd100 = load_product(PD100)
    tm510 = load_product(TM510)

    assert pd100.name.startswith("Преобразователь избыточного давления ОВЕН")
    assert pd100.manufacturer == "ОВЕН"
    assert len(pd100.specs) == 14
    assert tm510.specs["класс точности"] == "1,5; 1,0"


def test_broken_json_reports_line_and_column():
    with pytest.raises(InputError) as err:
        parse_product('{"name": "Манометр",\n "specs": {"диаметр": "100 мм",}\n}')

    assert "строка 2" in str(err.value)


def test_top_level_must_be_object():
    with pytest.raises(InputError, match="объект"):
        parse_product('[{"name": "Манометр"}]')


@pytest.mark.parametrize("payload", [{}, {"name": "   "}, {"name": None}])
def test_name_is_required(payload):
    with pytest.raises(InputError, match="name"):
        parse_product(json.dumps(payload))


def test_nested_spec_value_is_rejected():
    payload = {"name": "Манометр", "specs": {"диапазон": {"от": 0, "до": 1}}}

    with pytest.raises(InputError, match="диапазон"):
        parse_product(json.dumps(payload, ensure_ascii=False))


def test_specs_must_be_object():
    with pytest.raises(InputError, match="specs"):
        parse_product('{"name": "Манометр", "specs": ["IP40"]}')


def test_numbers_and_lists_in_specs_become_text():
    payload = {"name": "Манометр", "specs": {"класс точности": 1.5, "резьба": ["G1/2", "М20х1,5"]}}

    product = parse_product(json.dumps(payload, ensure_ascii=False))

    assert product.specs == {"класс точности": "1.5", "резьба": "G1/2; М20х1,5"}


def test_boolean_spec_value_is_rejected():
    with pytest.raises(InputError, match="взрывозащита"):
        parse_product('{"name": "Датчик", "specs": {"взрывозащита": true}}')


def test_empty_values_are_dropped_and_remembered():
    payload = {
        "name": "  Манометр ТМ-510  ",
        "manufacturer": " ",
        "specs": {"класс точности": "", "диаметр корпуса": "100 мм", "резьба": None},
        "description": "",
    }

    product = parse_product(json.dumps(payload, ensure_ascii=False))

    assert product.name == "Манометр ТМ-510"
    assert product.manufacturer is None
    assert product.description is None
    assert product.specs == {"диаметр корпуса": "100 мм"}
    assert product.empty_fields == ("manufacturer", "specs.класс точности", "specs.резьба", "description")


def test_repeated_spec_keys_are_merged_not_lost():
    text = '{"name": "Манометр", "specs": {"класс точности": "1,5", "класс точности": "1,0", "класс  точности ": "2,5"}}'

    product = parse_product(text)

    assert product.specs == {"класс точности": "1,5; 1,0; 2,5"}


def test_repeated_top_level_field_is_an_error():
    with pytest.raises(InputError, match="повторяется: name"):
        parse_product('{"name": "Манометр", "name": "Датчик"}')


def test_specs_null_and_unnamed_spec_are_reported():
    product = parse_product('{"name": "Манометр", "specs": null}')
    unnamed = parse_product('{"name": "Манометр", "specs": {"": "100 мм"}}')

    assert product.specs == {} and product.empty_fields == ("specs",)
    assert unnamed.specs == {} and unnamed.empty_fields == ("specs.(без названия)",)


def test_name_is_single_line():
    assert parse_product('{"name": "Манометр\\n# ТМ-510"}').name == "Манометр # ТМ-510"


def test_extra_fields_are_kept():
    product = parse_product('{"name": "Манометр", "source_url": "https://example.ru/tm"}')

    assert product.model_extra == {"source_url": "https://example.ru/tm"}


def test_missing_file_gives_input_error(tmp_path):
    with pytest.raises(InputError, match="не найден"):
        load_product(tmp_path / "net.json")
