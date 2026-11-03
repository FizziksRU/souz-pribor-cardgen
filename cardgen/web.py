"""Одностраничная веб-форма: python -m cardgen.web, затем http://127.0.0.1:8000"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from .llm import LLMConfigError, client_from_env
from .loader import InputError, parse_product
from .pipeline import generate_card
from .render import to_markdown

MAX_BODY = 200_000  # байт; карточка одного прибора умещается с большим запасом
INDEX = Path(__file__).with_name("static") / "index.html"

log = logging.getLogger(__name__)
load_dotenv()
app = FastAPI(title="Черновики карточек КИП", docs_url=None, redoc_url=None)
# Сервер слушает только 127.0.0.1, но чужая страница с подменой DNS (DNS rebinding) достучалась бы
# до него под своим именем и тратила бы деньги на модель — принимаем только локальные имена
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Необработанная ошибка на %s", request.url.path)
    return _error("Внутренняя ошибка сервера — подробности в журнале сервера.", 500)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX.read_text(encoding="utf-8")


@app.post("/api/card")
async def make_card(request: Request, dry_run: bool = False):
    # только application/json: «простые» запросы чужих страниц (text/plain) браузер шлёт
    # без предварительной проверки CORS, и они тратили бы кредиты модели
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        return _error("Нужен заголовок Content-Type: application/json.", 415)
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY:
        return _error(f"Слишком большой запрос: больше {MAX_BODY} байт.", 413)
    body = await request.body()
    if len(body) > MAX_BODY:
        return _error(f"Слишком большой запрос: больше {MAX_BODY} байт.", 413)
    try:
        product = parse_product(body.decode("utf-8-sig"))
    except UnicodeDecodeError:
        return _error("Тело запроса должно быть в кодировке UTF-8.", 422)
    except InputError as exc:
        return _error(str(exc), 422)
    try:
        client = None if dry_run else client_from_env()
    except LLMConfigError as exc:
        return _error(str(exc), 503)
    result = await run_in_threadpool(generate_card, product, client)
    return {"result": result.model_dump(mode="json"), "markdown": to_markdown(result)}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
