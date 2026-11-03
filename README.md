# GigaAM-Diarize: ASR + диаризация через HTTP API

Сервис распознавания речи (GigaAM от Sber) с диаризацией спикеров (pyannote),
завёрнутый в FastAPI и работающий в Docker-контейнере на GPU. Распознавание идёт
асинхронно: запрос создаёт задачу, результат забирается по её id.

## Возможности

- `POST /transcribe` — транскрибация аудио (опционально пословные таймстампы).
- `POST /transcribe_diarized` — транскрибация с разделением по спикерам.
- `GET /jobs/{id}` / `DELETE /jobs/{id}` — получение и удаление результата задачи.
- `GET /jobs` — список задач.
- `GET /health` — статус сервиса, версия API, загруженная модель, TTL задач.
- Автоудаление просроченных задач: результаты хранятся только в оперативной памяти
  и удаляются по истечении `JOB_TTL_SECONDS`.

## Требования
- Docker с [NVIDIA Container Toolkit](https://docs.nvidia.com/data-center/cloud-native/container-toolkit/).
- Docker Compose v2.
- Токен Hugging Face с доступом к гейтед-моделям pyannote (`pyannote/speaker-diarization-3.1`
  и `pyannote/segmentation-3.1`). Получить: https://huggingface.co/settings/tokens
  (нужно принять лицензии на страницах моделей).

## Сборка контейнера

1. Создайте файл `.env` в корне репозитория:

   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   # необязательные параметры:
   GIGAAM_MODEL=v3_e2e_rnnt
   API_PORT=8000
   JOB_TTL_SECONDS=1800
   ```

2. Соберите образ (compose сам передаст `HF_TOKEN` как build secret):

   ```bash
   docker compose build
   ```

   либо явно:

   ```bash
   docker compose build --secret id=hf_token,env=HF_TOKEN
   ```

   При сборке в контейнер скачивается чекпоинт выбранной модели
   (`prepare_models.py`), поэтому собранный образ работает без интернета.

3. Запустите:

   ```bash
   docker compose up -d
   ```

   Сервис поднимется на `http://localhost:${API_PORT:-8000}`. Готовность:
   `GET /health` (healthcheck в compose опрашивает его каждые 30 секунд).

## Параметры, на которые стоит обратить внимание

| Параметр | Где задаётся | По умолчанию | Описание |
|---|---|---|---|
| `HF_TOKEN` | `.env` (секрет сборки и runtime) | — | Токен HF для гейтед-моделей pyannote. Без него диаризация не сможет скачать модели. |
| `GIGAAM_MODEL` | build ARG + runtime env | `v3_e2e_rnnt` | Какой чекпоинт GigaAM скачать при сборке и загрузить при старте. |
| `API_PORT` | `.env` | `8000` | Порт на хосте, который проксируется на 8000 внутри контейнера. |
| `JOB_TTL_SECONDS` | runtime env | `1800` | Сколько секунд хранить завершённую задачу, потом автоудаление. `0` — отключить автоудаление. |

Доступные варианты `GIGAAM_MODEL`: `v1_ctc`, `v1_rnnt`, `v2_ctc`, `v2_rnnt`,
`v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`, `v3_e2e_rnnt`, `multilingual_ctc`,
`multilingual_large_ctc`. Для API по умолчанию рекомендуются `v3_e2e_rnnt`
(лучшее качество) или `v3_e2e_ctc` (быстрее, проще постобработка). При смене
модели не забудьте пересобрать образ — чекпоинт запекается на этапе сборки.

## Примеры использования

Транскрибация:

```bash
curl -s -X POST "http://localhost:8000/transcribe" -F "file=@audio.wav" \
  | jq '{job_id, status}'
```

Диаризация с фиксированным числом спикеров и пословными таймстампами:

```bash
curl -s -X POST "http://localhost:8000/transcribe_diarized" \
  -F "file=@audio.wav" -F "num_speakers=2" \
  | jq '{job_id, status}'
```

Получение результата (повторять, пока `status != "completed"`):

```bash
curl -s http://localhost:8000/jobs/<job_id> | jq
```

Только один запрос распознаётся одновременно: если модель занята, второй POST
вернёт `409 Conflict`. Очередь не копится — повторите запрос позже.