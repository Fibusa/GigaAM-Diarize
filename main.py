"""
GigaAM ASR API — отдельный сервер FastAPI.

Эндпоинты:
    GET    /health              — проверка состояния сервера
    POST   /transcribe          — транскрибация: ставит задачу в очередь, возвращает job_id
    POST   /transcribe_diarized — транскрибация + диаризация: очередь, возвращает job_id

Отслеживание задач (jobs):
    GET    /jobs                — список задач
    GET    /jobs/{job_id}       — статус и результат задачи
    DELETE /jobs/{job_id}       — отменить (queued) или удалить задачу

Завершённые (completed/failed) задачи автоматически удаляются через
JOB_TTL_SECONDS секунд после окончания (по умолчанию 1800; 0 — выключить).
Настройка передаётся переменной окружения JOB_TTL_SECONDS.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

import gigaam
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from gigaam.diarize_utils import assign_speakers
from pydantic import BaseModel

if TYPE_CHECKING:
    from gigaam.types import LongformTranscriptionResult

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Автоудаление завершённых задач: сколько секунд результат доступен после
# окончания задачи (completed/failed). 0 или отрицательное — выключить.
# Активные (queued/running) задачи никогда не удаляются автоматически.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "1800"))

# ---------------------------------------------------------------------------
# Глобальное состояние (модель загружается лениво при первом запросе)
# ---------------------------------------------------------------------------
asr_model = None


# ---------------------------------------------------------------------------
# Pydantic схемы
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    device: str
    job_ttl_seconds: Optional[int] = None  # None = автоудаление выключено


class WordTimestamp(BaseModel):
    text: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    text: str
    start: float
    end: float
    words: Optional[list[WordTimestamp]] = None


class TranscriptionResponse(BaseModel):
    text: str
    segments: list[TranscriptionSegment]
    duration: Optional[float] = None


class Utterance(BaseModel):
    """Реплика одного спикера: фраза целиком, без разбивки на слова."""

    start: float
    end: float
    speaker: Optional[str] = None
    text: str


class TranscribeDiarizedResponse(BaseModel):
    text: str
    utterances: list[Utterance]
    speakers: list[str]
    num_speakers: int
    duration: Optional[float] = None


class JobMode(str, Enum):
    """Тип задачи: какой из базовых эндпоинтов выполняется в фоне."""

    transcribe = "transcribe"
    transcribe_diarized = "transcribe_diarized"


class JobStatus(str, Enum):
    queued = "queued"          # в очереди, ещё не начала выполняться
    running = "running"        # выполняется прямо сейчас
    completed = "completed"    # успешно завершена, результат доступен
    failed = "failed"          # завершилась с ошибкой
    cancelled = "cancelled"    # отменена до начала выполнения


class JobInfo(BaseModel):
    """Статус задачи без результата."""

    job_id: str
    mode: JobMode
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class JobResponse(JobInfo):
    """Статус задачи вместе с результатом (когда он есть)."""

    result: Optional[
        Union[TranscriptionResponse, TranscribeDiarizedResponse]
    ] = None


class JobListResponse(BaseModel):
    jobs: list[JobInfo]


# ---------------------------------------------------------------------------
# FastAPI приложение
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GigaAM ASR API",
    description=(
        "API для транскрибации длинного аудио и диаризации спикеров с помощью GigaAM"
    ),
    version="1.6.0",
)


# ---------------------------------------------------------------------------
# Служебные функции
# ---------------------------------------------------------------------------
def _ensure_model_loaded():
    """Ленивая загрузка модели при первом запросе."""
    global asr_model
    if asr_model is None:
        asr_model = gigaam.load_model(
            DEFAULT_MODEL,
            device=torch.device(DEVICE),
        )


# ---------------------------------------------------------------------------
# Фоновые задачи (jobs)
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _JobRecord:
    """Внутреннее представление фоновой задачи."""

    job_id: str
    mode: JobMode
    tmp_path: str
    params: dict
    created_at: str
    status: JobStatus = JobStatus.queued
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[BaseModel] = None
    future: Optional[Future] = None


# Один воркер: GPU-инференс не должен выполняться параллельно (мало VRAM),
# остальные задачи ждут в очереди.
_jobs_lock = threading.Lock()
_jobs: dict[str, _JobRecord] = {}
_job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gigaam-job")

# Как часто фоновый уборщик проверяет просроченные задачи.
_JOB_SWEEP_INTERVAL_SECONDS = 60


def _sweep_expired_jobs() -> int:
    """Удаляет завершённые задачи, у которых finished_at старше JOB_TTL_SECONDS.

    Возвращает число удалённых задач. Активные (queued/running) задачи
    не трогаются: у них нет результата, который стоило бы освобождать.
    """
    if JOB_TTL_SECONDS <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=JOB_TTL_SECONDS)
    expired: list[str] = []
    with _jobs_lock:
        for job_id, job in _jobs.items():
            if job.status not in (JobStatus.completed, JobStatus.failed):
                continue
            if not job.finished_at:
                continue
            try:
                finished_at = datetime.fromisoformat(job.finished_at)
            except ValueError:
                continue
            if finished_at <= cutoff:
                expired.append(job_id)
        for job_id in expired:
            del _jobs[job_id]
    if expired:
        print(f"[jobs] Автоудалено просроченных задач: {len(expired)}")
    return len(expired)


def _sweep_loop() -> None:
    """Цикл фонового уборщика: раз в JOB_SWEEP_INTERVAL секунд чистит jobs."""
    while True:
        time.sleep(_JOB_SWEEP_INTERVAL_SECONDS)
        try:
            _sweep_expired_jobs()
        except Exception:  # noqa: BLE001 — уборщик не должен ронять сервер
            pass


threading.Thread(
    target=_sweep_loop,
    daemon=True,
    name="gigaam-job-sweeper",
).start()


def _execute_job(job_id: str) -> None:
    """Выполняет задачу из очереди. Запускается в единственном воркере."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job.status is not JobStatus.queued:
            return  # задачу успели отменить/удалить до старта
        job.status = JobStatus.running
        job.started_at = _now_iso()

    try:
        _ensure_model_loaded()
        runner = _MODE_RUNNERS[job.mode]
        result = runner(job.tmp_path, **job.params)
    except Exception as exc:  # noqa: BLE001 — ошибка задачи, а не сервера
        with _jobs_lock:
            if job.status is JobStatus.running:
                job.status = JobStatus.failed
                job.finished_at = _now_iso()
                job.error = str(exc) or exc.__class__.__name__
        _remove_tmp(job.tmp_path)
        job.tmp_path = ""
        return

    with _jobs_lock:
        if job.status is JobStatus.running:
            job.status = JobStatus.completed
            job.finished_at = _now_iso()
            job.result = result
    _remove_tmp(job.tmp_path)
    job.tmp_path = ""


def _validate_speaker_range(
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    """Общая 422-проверка для задач, связанных с диаризацией."""
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise HTTPException(
            status_code=422,
            detail="min_speakers не может быть больше max_speakers",
        )


def _get_job_or_404(job_id: str) -> _JobRecord:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Задача {job_id} не найдена (возможно, уже удалена)",
        )
    return job


def _check_model_free() -> None:
    """Поднимает 409, если модель занята: есть queued/running задача."""
    with _jobs_lock:
        busy = any(
            job.status in (JobStatus.queued, JobStatus.running)
            for job in _jobs.values()
        )
    if busy:
        raise HTTPException(
            status_code=409,
            detail=(
                "Модель занята: уже выполняется или ожидает другая задача. "
                "Дождитесь её завершения (GET /jobs) и повторите запрос"
            ),
        )


def _job_to_response(job: _JobRecord) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        mode=job.mode,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        result=job.result,
    )


async def _save_upload_to_tmp(file: UploadFile) -> str:
    """Сохраняет загруженный файл во временный файл и возвращает его путь."""
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
    finally:
        tmp.close()
    return tmp.name


def _remove_tmp(path: Optional[str]) -> None:
    """Удаляет временный файл, не маскируя возможные исключения выше."""
    if not path:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def _audio_duration(path: str) -> Optional[float]:
    """Длительность аудио в секундах (None, если файл не удалось прочитать)."""
    try:
        return float(sf.info(path).duration)
    except Exception:
        return None


def _run_transcribe_diarized(
    model: "gigaam.GigaAMASR",
    wav_path: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    min_overlap: float,
) -> tuple["LongformTranscriptionResult", Optional[float]]:
    """Синхронная транскрибация + диаризация: вызывается в отдельном потоке."""
    annotation = model.diarize(
        wav_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    result = model.transcribe_longform(wav_path, word_timestamps=True)
    labeled = assign_speakers(result, annotation, min_overlap=min_overlap)
    return labeled, _audio_duration(wav_path)


def _build_utterances(
    labeled: "LongformTranscriptionResult",
) -> list[Utterance]:
    """
    Склеивает слова с метками спикеров в реплики: идём по словам подряд и
    начинаем новую реплику, когда меняется спикер («следующая фраза
    начинается, когда говорит следующий спикер»).

    Слова без метки спикера (None) присоединяются к текущей реплике,
    а если её нет — образуют отдельную реплику с speaker=None.
    """
    utterances: list[Utterance] = []
    current_speaker: Optional[str] = None
    current_words: list["Word"] = []

    def _flush() -> None:
        if not current_words:
            return
        utterances.append(
            Utterance(
                start=round(current_words[0].start, 3),
                end=round(current_words[-1].end, 3),
                speaker=current_speaker,
                text=" ".join(w.text for w in current_words),
            )
        )

    for seg in labeled:
        for w in seg.words or []:
            if current_words and w.speaker != current_speaker:
                _flush()
                current_words = []
            current_speaker = w.speaker
            current_words.append(w)
    _flush()

    return utterances


# ---------------------------------------------------------------------------
# Синхронные обработчики (общие для эндпоинтов и фоновых задач)
# ---------------------------------------------------------------------------
def _transcribe_sync(
    wav_path: str,
    word_timestamps: bool,
) -> TranscriptionResponse:
    """Синхронная транскрибация длинного аудио."""
    result = asr_model.transcribe_longform(
        wav_path,
        word_timestamps=word_timestamps,
    )

    segments = []
    for seg in result:
        seg_words = None
        if seg.words:
            seg_words = [
                WordTimestamp(text=w.text, start=w.start, end=w.end)
                for w in seg.words
            ]
        segments.append(
            TranscriptionSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                words=seg_words,
            )
        )

    return TranscriptionResponse(
        text=result.text,
        segments=segments,
        duration=_audio_duration(wav_path),
    )


def _transcribe_diarized_sync(
    wav_path: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    min_overlap: float,
) -> TranscribeDiarizedResponse:
    """Синхронная транскрибация + диаризация за один проход."""
    result, duration = _run_transcribe_diarized(
        asr_model,
        wav_path,
        num_speakers,
        min_speakers,
        max_speakers,
        min_overlap,
    )
    utterances = _build_utterances(result)
    speakers = list(
        dict.fromkeys(u.speaker for u in utterances if u.speaker is not None)
    )
    return TranscribeDiarizedResponse(
        text=result.text,
        utterances=utterances,
        speakers=speakers,
        num_speakers=len(speakers),
        duration=duration,
    )


# Регистрация обработчиков для фоновых задач.
_MODE_RUNNERS = {
    JobMode.transcribe: _transcribe_sync,
    JobMode.transcribe_diarized: _transcribe_diarized_sync,
}


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Проверка состояния сервера."""
    return HealthResponse(
        status="ok",
        device=DEVICE,
        job_ttl_seconds=JOB_TTL_SECONDS if JOB_TTL_SECONDS > 0 else None,
    )


@app.post("/transcribe", response_model=JobInfo, status_code=202)
async def transcribe(
    file: UploadFile = File(...),
    word_timestamps: bool = Query(
        False, description="Возвращать ли тайминги отдельных слов"
    ),
):
    """
    Ставит транскрибацию длинного аудио в очередь и сразу возвращает job_id.

    Если модель уже занята другой задачей (queued/running), возвращается 409.
    Статус и результат — GET /jobs/{job_id}: когда status=completed, в result
    лежит ответ того же формата, что раньше отдавал синхронный /transcribe
    (text, segments, duration).
    """
    _check_model_free()
    tmp_path = await _save_upload_to_tmp(file)
    job_id = uuid.uuid4().hex
    job = _JobRecord(
        job_id=job_id,
        mode=JobMode.transcribe,
        tmp_path=tmp_path,
        params=dict(word_timestamps=word_timestamps),
        created_at=_now_iso(),
    )
    with _jobs_lock:
        _jobs[job_id] = job
    job.future = _job_executor.submit(_execute_job, job_id)
    return JobInfo(
        job_id=job.job_id,
        mode=job.mode,
        status=job.status,
        created_at=job.created_at,
    )


@app.post("/transcribe_diarized", response_model=JobInfo, status_code=202)
async def transcribe_diarized(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Query(
        None, ge=1, description="Точное число спикеров"
    ),
    min_speakers: Optional[int] = Query(
        None, ge=1, description="Минимальное число спикеров"
    ),
    max_speakers: Optional[int] = Query(
        None, ge=1, description="Максимальное число спикеров"
    ),
    min_overlap: float = Query(
        0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Минимальная доля длительности слова, покрываемая тёрном спикера, "
            "чтобы слово получило его метку"
        ),
    ),
):
    """
    Ставит транскрибацию + диаризацию в очередь и сразу возвращает job_id.

    Если модель уже занята другой задачей (queued/running), возвращается 409.
    Статус и результат — GET /jobs/{job_id}: когда status=completed, в result
    лежит text, utterances (реплики по спикерам), speakers, num_speakers,
    duration.
    """
    _validate_speaker_range(min_speakers, max_speakers)
    _check_model_free()
    tmp_path = await _save_upload_to_tmp(file)
    job_id = uuid.uuid4().hex
    job = _JobRecord(
        job_id=job_id,
        mode=JobMode.transcribe_diarized,
        tmp_path=tmp_path,
        params=dict(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            min_overlap=min_overlap,
        ),
        created_at=_now_iso(),
    )
    with _jobs_lock:
        _jobs[job_id] = job
    job.future = _job_executor.submit(_execute_job, job_id)
    return JobInfo(
        job_id=job.job_id,
        mode=job.mode,
        status=job.status,
        created_at=job.created_at,
    )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs():
    """Список всех задач (без результатов), отсортированный по времени создания."""
    with _jobs_lock:
        records = sorted(_jobs.values(), key=lambda job: job.created_at)
    return JobListResponse(
        jobs=[
            JobInfo(
                job_id=job.job_id,
                mode=job.mode,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                error=job.error,
            )
            for job in records
        ]
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Статус задачи; для completed — готовый результат того же формата, что у соответствующего синхронного эндпоинта."""
    return _job_to_response(_get_job_or_404(job_id))


@app.delete("/jobs/{job_id}", response_model=JobResponse)
async def delete_job(job_id: str):
    """
    Отменяет задачу в очереди (queued → cancelled) или удаляет завершённую.
    Для running задачи вернёт 409 — дождитесь завершения.
    """
    job = _get_job_or_404(job_id)
    with _jobs_lock:
        if job.status is JobStatus.running:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Задача выполняется и не может быть отменена; "
                    "дождитесь завершения"
                ),
            )
        if job.status is JobStatus.queued:
            job.status = JobStatus.cancelled
            job.finished_at = _now_iso()
        response = _job_to_response(job)
        del _jobs[job_id]
    if job.tmp_path:
        _remove_tmp(job.tmp_path)
        job.tmp_path = ""
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
