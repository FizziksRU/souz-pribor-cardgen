from cardgen.loader import load_product, product_from_dict
from cardgen.pipeline import generate_card
from cardgen.render import to_markdown
from tests.conftest import PD100, TM510, card_json


def test_ok_card_has_every_block(fake_llm):
    llm = fake_llm(card_json(data_issues=["Погрешность есть только в обозначении модели"]))
    result = generate_card(load_product(PD100), client=llm)

    md = to_markdown(result)

    assert md.startswith("# Преобразователь избыточного давления ОВЕН ПД100-ДИ1,6-111-0,5")
    assert f"## SEO-заголовок ({len(result.card.seo_title)} симв.)" in md
    assert "| Производитель | ОВЕН |" in md
    assert "| Класс точности / погрешность | **данных нет** |" in md
    assert "- датчик давления ОВЕН ПД100" in md
    assert "**Кому подходит**" in md and "**Какие задачи решает**" in md
    assert "Погрешность есть только в обозначении модели" in md


def test_dry_run_says_texts_are_not_generated():
    md = to_markdown(generate_card(load_product(TM510), client=None))

    assert "режим без модели" in md
    assert "## SEO-заголовок" not in md
    assert "1,5; 1,0" in md  # блок «Требует уточнения» есть и без модели


def test_needs_review_lists_check_errors(fake_llm):
    bad = card_json(seo_title="Датчик на 2,5 МПа")
    md = to_markdown(generate_card(load_product(PD100), client=fake_llm(bad, bad)))

    assert "## Замечания проверки" in md
    assert "«2,5 МПа»" in md


def test_table_cells_are_escaped():
    product = product_from_dict({"name": "Прибор", "specs": {"шкала": "0|10\nбар"}})

    md = to_markdown(generate_card(product, client=None))

    assert "| Шкала | 0\\|10 бар |" in md


def test_error_is_shown(fake_llm, llm_error):
    md = to_markdown(generate_card(load_product(PD100), client=fake_llm(llm_error)))

    assert "Сервис LLM недоступен" in md
