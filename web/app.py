"""
Веб-интерфейс GigaAM: статика + прокси к API-серверу диаризации.

Отдельный сервис (контейнер `web`), который общается с контейнером `gigaam-api`
по внутренней сети compose. Браузер открывает только этот сервис, поэтому
CORS не требуется: все запросы идут на один origin, а прокси пересылает их
в API-сервер.

Зависимости — только те же пакеты, что в extra "api" pyproject.toml
(fastapi/starlette, python-multipart, uvicorn); проксирование выполняется
стандартным http.client со стримингом multipart-тела, поэтому большие
аудио-/видеофайлы не буферизуются в память.

Локальный запуск (рядом с API: uvicorn main:app --port 8000):
    uvicorn app:app --host 0.0.0.0 --port 8001
    или: python app.py

Переменные окружения:
    API_BASE_URL  адрес API-сервера (по умолчанию http://localhost:8000,
                  в контейнере — http://gigaam-api:8000)
    WEBUI_PORT    порт веб-интерфейса  (по умолчанию 8001)
"""

import json
import os
import uuid
from http.client import HTTPConnection, HTTPSConnection
from typing import Iterator, Optional, Tuple
from urllib.parse import urlencode, urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
WEBUI_PORT = int(os.getenv("WEBUI_PORT", "8001"))

_STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

_HEALTH_TIMEOUT = 10.0
_SHORT_TIMEOUT = 60.0  # статусы, список задач, отмена
_UPLOAD_TIMEOUT = 3600.0  # POST с файлом: загрузка и сохранение могут идти долго
_CHUNK_SIZE = 1024 * 1024

# Параметры запроса, которые разрешено пробрасывать в API-сервер.
_FORWARD_QUERY = {
    "num_speakers",
    "min_speakers",
    "max_speakers",
    "min_overlap",
    "word_timestamps",
}

app = FastAPI(title="GigaAM Web UI", version="1.0.0", docs_url=None, redoc_url=None)


class ProxyError(Exception):
    """API-сервер недоступен или ответил ошибкой транспорта."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _proxy_request(
    method: str,
    path: str,
    query: str = "",
    body=None,
    headers: Optional[dict] = None,
    timeout: float = _SHORT_TIMEOUT,
) -> Response:
    """Один вызов к API-серверу; ответ возвращается «как есть»."""
    url = urlparse(API_BASE_URL)
    conn_cls = HTTPSConnection if url.scheme == "https" else HTTPConnection
    conn = conn_cls(url.netloc, timeout=timeout)
    try:
        target = f"{path}?{query}" if query else path
        # body-генератор отправляется chunked-кодировкой: файл течёт с диска
        # на сервер без буферизации в памяти.
        conn.request(method, target, body=body, headers=headers or {}, encode_chunked=True)
        resp = conn.getresponse()
        data = resp.read()
        return Response(
            content=data,
            status_code=resp.status,
            media_type=resp.getheader("content-type") or "application/json",
        )
    except (OSError, TimeoutError) as exc:
        # ConnectionRefusedError, socket.timeout и прочие ошибки транспорта.
        raise ProxyError(
            502, f"API-сервер ({API_BASE_URL}) недоступен: {exc}"
        ) from exc
    finally:
        conn.close()


def _proxy_error(exc: ProxyError) -> Response:
    return Response(
        content=json.dumps({"detail": exc.detail}, ensure_ascii=False),
        status_code=exc.status,
        media_type="application/json; charset=utf-8",
    )


def _escape_filename(name: str) -> str:
    for ch in ("\\", '"', "\r", "\n"):
        name = name.replace(ch, "_")
    return name


def _multipart_body(
    filename: str, content_type: Optional[str], file: UploadFile
) -> Tuple[str, Iterator[bytes]]:
    """Собирает multipart/form-data поток байтов с содержимым файла."""
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{_escape_filename(filename)}"\r\n'
        f"Content-Type: {content_type or 'application/octet-stream'}\r\n"
        "\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")

    def stream() -> Iterator[bytes]:
        yield head
        try:
            while True:
                chunk = file.file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            file.close()

    return boundary, stream()


def _proxy_upload(request: Request, file: UploadFile, backend_path: str) -> Response:
    query = urlencode(
        [(k, v) for k, v in request.query_params.items() if k in _FORWARD_QUERY]
    )
    boundary, body = _multipart_body(file.filename or "file", file.content_type, file)
    try:
        return _proxy_request(
            "POST",
            backend_path,
            query=query,
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=_UPLOAD_TIMEOUT,
        )
    except ProxyError as exc:
        return _proxy_error(exc)


def _clean_job_id(job_id: str) -> str:
    if not job_id or len(job_id) > 64 or any(
        c not in "0123456789abcdef" for c in job_id.lower()
    ):
        raise HTTPException(status_code=400, detail="Некорректный job_id")
    return job_id


# ---------------------------------------------------------------------------
# Проксирование API-сервера
# ---------------------------------------------------------------------------
@app.post("/api/transcribe_diarized")
def proxy_transcribe_diarized(request: Request, file: UploadFile = File(...)) -> Response:
    """Загрузка файла на транскрибацию + диаризацию (прокси на /transcribe_diarized)."""
    return _proxy_upload(request, file, "/transcribe_diarized")


@app.post("/api/transcribe")
def proxy_transcribe(request: Request, file: UploadFile = File(...)) -> Response:
    """Загрузка файла на транскрибацию без диаризации (прокси на /transcribe)."""
    return _proxy_upload(request, file, "/transcribe")


@app.get("/api/health")
def proxy_health() -> Response:
    try:
        return _proxy_request("GET", "/health", timeout=_HEALTH_TIMEOUT)
    except ProxyError as exc:
        return _proxy_error(exc)


@app.get("/api/jobs")
def proxy_jobs() -> Response:
    try:
        return _proxy_request("GET", "/jobs")
    except ProxyError as exc:
        return _proxy_error(exc)


@app.get("/api/jobs/{job_id}")
def proxy_job(job_id: str) -> Response:
    try:
        return _proxy_request("GET", f"/jobs/{_clean_job_id(job_id)}")
    except ProxyError as exc:
        return _proxy_error(exc)


@app.delete("/api/jobs/{job_id}")
def proxy_delete_job(job_id: str) -> Response:
    try:
        return _proxy_request("DELETE", f"/jobs/{_clean_job_id(job_id)}")
    except ProxyError as exc:
        return _proxy_error(exc)


# ---------------------------------------------------------------------------
# Статический интерфейс
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=WEBUI_PORT)
