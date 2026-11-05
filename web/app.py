"""
Веб-интерфейс GigaAM: статика + прокси к API-серверу диаризации.

Отдельный сервис (контейнер `web`), который общается с контейнером `gigaam-api`
по внутренней сети compose. Браузер открывает только этот сервис, поэтому
CORS не требуется: все запросы идут на один origin, а прокси пересылает их
в API-сервер.

Зависимости — те же пакеты, что в extra "api" pyproject.toml
(fastapi/starlette, python-multipart, uvicorn) плюс системный ffmpeg;
проксирование выполняется стандартным http.client со стримингом
multipart-тела, поэтому большие аудио-/видеофайлы не буферизуются в память.

Перед отправкой в API загрузка перекодируется ffmpeg'ом в 16 кГц mono WAV
(родная частота моделей GigaAM). Контейнерные форматы (MP4/M4A/WebM), у
которых аудиопоток начинается с ненулевого start_time (edit list — типично
для записей экрана и телефонов), иначе ломают декодирование 10-секундных
окон в pyannote: torchcodec возвращает меньше сэмплов, чем ожидает
Audio.crop, и задача падает с ошибкой вида
"resulted in N samples instead of the expected M samples".

Локальный запуск (рядом с API: uvicorn main:app --port 8000):
    uvicorn app:app --host 0.0.0.0 --port 8001
    или: python app.py

Переменные окружения:
    API_BASE_URL  адрес API-сервера (по умолчанию http://localhost:8000,
                  в контейнере — http://gigaam-api:8000)
    WEBUI_PORT    порт веб-интерфейса  (по умолчанию 8001)
    OPENROUTER_API_KEY   ключ OpenRouter для суммаризации встреч (без него
                  POST /api/summarize возвращает 503)
    OPENROUTER_MODEL     модель OpenRouter по умолчанию
                  (по умолчанию z-ai/glm-5.3-flash)
"""

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from http.client import HTTPConnection, HTTPSConnection
from typing import Iterator, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
WEBUI_PORT = int(os.getenv("WEBUI_PORT", "8001"))

_STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

_HEALTH_TIMEOUT = 10.0
_SHORT_TIMEOUT = 60.0  # статусы, список задач, отмена
_UPLOAD_TIMEOUT = 3600.0  # POST с файлом: загрузка и сохранение могут идти долго
_CHUNK_SIZE = 1024 * 1024
_CONVERT_TIMEOUT = 900.0  # максимум на перекодирование ffmpeg, сек
_FFMPEG = os.getenv("FFMPEG", "ffmpeg")

# Параметры запроса, которые разрешено пробрасывать в API-сервер.
_FORWARD_QUERY = {
    "num_speakers",
    "min_speakers",
    "max_speakers",
    "min_overlap",
    "word_timestamps",
}

# ---------------------------------------------------------------------------
# OpenRouter: суммаризация встреч через LLM
# ---------------------------------------------------------------------------
_OPENROUTER_HOST = "openrouter.ai"
_OPENROUTER_PATH = "/api/v1/chat/completions"
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")
_SUMMARIZE_TIMEOUT = 300.0  # максимум на один запрос к LLM, сек

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


def _silent_unlink(path: Optional[str]) -> None:
    """Удаляет временный файл, игнорируя ошибки (уборка в finally)."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _convert_to_wav_16k_mono(src_path: str) -> str:
    """Перекодирует src в 16 кГц mono WAV через ffmpeg, возвращает путь к WAV.

    Бросает HTTPException (422/504/500), если файл не декодируется, ffmpeg
    не успел или его нет в системе. Вызывающий код отвечает за удаление
    результата.
    """
    dst = tempfile.NamedTemporaryFile(
        delete=False, prefix="converted-", suffix=".wav"
    )
    dst.close()
    cmd = [
        _FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", src_path,
        "-vn", "-sn", "-dn",          # без видео, субтитров и потоков данных
        "-ac", "1", "-ar", "16000",   # mono, 16 кГц — формат моделей GigaAM
        "-f", "wav", dst.name,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_CONVERT_TIMEOUT
        )
    except subprocess.TimeoutExpired as exc:
        _silent_unlink(dst.name)
        raise HTTPException(
            status_code=504,
            detail=(
                f"ffmpeg не уложился в {_CONVERT_TIMEOUT:.0f} с "
                "при перекодировании файла"
            ),
        ) from exc
    except OSError as exc:  # бинарник ffmpeg не найден
        _silent_unlink(dst.name)
        raise HTTPException(
            status_code=500, detail=f"ffmpeg недоступен: {exc}"
        ) from exc
    if proc.returncode != 0:
        _silent_unlink(dst.name)
        stderr_tail = [
            line
            for line in proc.stderr.decode("utf-8", "replace").strip().splitlines()
            if line
        ][-3:]
        detail = "; ".join(stderr_tail) or "неизвестная ошибка ffmpeg"
        raise HTTPException(
            status_code=422,
            detail=f"Не удалось перекодировать файл в WAV: {detail}",
        )
    return dst.name


def _multipart_body(
    filename: str, content_type: Optional[str], fileobj
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
                chunk = fileobj.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
            yield tail
        finally:
            # Закрываем переданный файловый объект, когда API дочитал тело.
            fileobj.close()

    return boundary, stream()


def _proxy_upload(request: Request, file: UploadFile, backend_path: str) -> Response:
    """Загрузку → временный файл → ffmpeg → 16 кГц mono WAV → стриминг в API.

    Нормализация к WAV выполняется до отправки: pyannote в API-контейнере
    читает файлы через torchcodec, и MP4 с edit list сдвигом аудиопотока
    даёт дефицит сэмплов в 10-секундных окнах (задание падает). WAV лишён
    этого класса проблем; заодно API начинает корректно считать duration.
    """
    query = urlencode(
        [(k, v) for k, v in request.query_params.items() if k in _FORWARD_QUERY]
    )
    src_path: Optional[str] = None
    wav_path: Optional[str] = None
    try:
        # 1. Сохраняем загрузку как есть (суффикс сохраняем для ясности;
        # формат ffmpeg определяет по содержимому, не по расширению).
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        with tempfile.NamedTemporaryFile(
            delete=False, prefix="upload-", suffix=suffix
        ) as src:
            shutil.copyfileobj(file.file, src, _CHUNK_SIZE)
            src_path = src.name
        file.file.close()

        # 2. Нормализация: 16 кГц mono WAV.
        wav_path = _convert_to_wav_16k_mono(src_path)

        # 3. Отправляем WAV в API тем же стримингом multipart.
        base = os.path.splitext(os.path.basename(file.filename or ""))[0]
        wav_name = f"{base or 'upload'}.wav"
        boundary, body = _multipart_body(
            wav_name, "audio/wav", open(wav_path, "rb")
        )
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
    finally:
        _silent_unlink(src_path)
        _silent_unlink(wav_path)


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
# Суммаризация встречи через OpenRouter
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "Ты — ассистент, который суммаризирует расшифровки встреч (митингов) "
    "на русском языке. Тебе даётся стенограмма с метками времени и спикеров.\n"
    "\n"
    "Составь структурированное резюме встречи:\n"
    "1. Краткое описание (2–4 предложения): о чём встреча и кто участвовал.\n"
    "2. Основные темы обсуждения — короткий маркированный список.\n"
    "3. Принятые решения — список; если решений нет, так и напиши.\n"
    "4. Открытые вопросы и неясности.\n"
    "5. Action items (задачи): кто — что делает — к какому сроку; "
    "если срок не прозвучал, укажи «срок не назван».\n"
    "\n"
    "Правила:\n"
    "- Опирайся только на текст расшифровки, ничего не додумывай.\n"
    "- Сохраняй обозначения спикеров как в стенограмме (SPEAKER_XX), "
    "если настоящие имена не прозвучали.\n"
    "- Пиши кратко и по делу, без воды.\n"
    "- Отвечай на русском языке; если стенограмма на другом языке — "
    "отвечай на языке стенограммы.\n"
    "- Оформи ответ в Markdown с заголовками разделов."
)


class SummarizeRequest(BaseModel):
    """Тело запроса /api/summarize (все поля опциональны)."""

    utterances: Optional[List[dict]] = None
    text: Optional[str] = None
    duration: Optional[float] = None
    speakers: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None


def _fmt_llm_ts(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_llm_user_content(req: SummarizeRequest) -> str:
    """Собирает текст запроса к LLM из результата диаризации."""
    lines: List[str] = []
    if req.duration:
        lines.append(f"Длительность встречи: ~{_fmt_llm_ts(req.duration)}.")
    if req.speakers:
        lines.append("Спикеры: " + ", ".join(req.speakers) + ".")
    lines.append("Расшифровка диалога:")
    lines.append("")
    used = False
    for utt in req.utterances or []:
        if not isinstance(utt, dict):
            continue
        text = str(utt.get("text", "")).strip()
        if not text:
            continue
        start, end = utt.get("start"), utt.get("end")
        stamp = ""
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            stamp = f"[{_fmt_llm_ts(start)} – {_fmt_llm_ts(end)}] "
        lines.append(f"{stamp}{utt.get('speaker', 'SPEAKER')}: {text}")
        used = True
    if not used:
        lines.append((req.text or "").strip() or "(расшифровка пуста)")
    return "\n".join(lines)


@app.post("/api/summarize")
def summarize(req: SummarizeRequest) -> Response:
    """Отправляет расшифровку в OpenRouter и возвращает резюме встречи.

    Ключ OPENROUTER_API_KEY живёт только на сервере (в окружении
    web-контейнера) и никогда не передаётся в браузер.
    """
    if not _OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "OPENROUTER_API_KEY не задан: добавьте ключ в .env "
                "(строка OPENROUTER_API_KEY=...) и пересоздайте web-контейнер"
            ),
        )
    system_prompt = (req.system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    model = (req.model or "").strip() or _OPENROUTER_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_llm_user_content(req)},
        ],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "Authorization": f"Bearer {_OPENROUTER_API_KEY}",
        "X-Title": "GigaAM Web UI",
    }
    try:
        conn = HTTPSConnection(_OPENROUTER_HOST, timeout=_SUMMARIZE_TIMEOUT)
        try:
            conn.request("POST", _OPENROUTER_PATH, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            conn.close()
    except (OSError, TimeoutError) as exc:
        raise HTTPException(
            status_code=502, detail=f"OpenRouter недоступен: {exc}"
        ) from exc

    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        data = None

    if resp.status != 200:
        # Прозрачно пробрасываем статус и текст ошибки OpenRouter.
        detail = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err)
            elif err is not None:
                detail = str(err)
        if not detail:
            detail = raw.decode("utf-8", "replace")[:300]
        raise HTTPException(
            status_code=resp.status,
            detail=f"OpenRouter вернул {resp.status}: {detail}",
        )
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="OpenRouter вернул не-JSON ответ")
    choices = data.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
    if not content.strip():
        raise HTTPException(status_code=502, detail="OpenRouter вернул пустой ответ модели")
    return Response(
        content=json.dumps(
            {
                "summary": content,
                "model": data.get("model") or model,
                "usage": data.get("usage"),
            },
            ensure_ascii=False,
        ),
        media_type="application/json; charset=utf-8",
    )


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
