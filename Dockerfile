# syntax=docker/dockerfile:1

# Тот же базовый образ: CUDA 13.0 + cuDNN 9, torch предустановлен.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-devel

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Системные зависимости для работы с аудио.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости отдельным слоем: пересобираются только
# при изменении pyproject.toml, а не при каждом изменении кода.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --break-system-packages ".[api,longform]" \
    && pip install --no-cache-dir --break-system-packages \
        torchcodec --index-url https://download.pytorch.org/whl/cu130

# Код приложения.
COPY gigaam ./gigaam
COPY main.py prepare_models.py ./

# Слой предзагрузки моделей: чекпоинт GigaAM и пайплайны pyannote
# скачиваются в кэш образа, чтобы первый запрос не ждал скачивания.
# Аргументы объявлены здесь (после слоёв с зависимостями), поэтому
# смена токена/модели не инвалидирует кэш предыдущих слоёв.
ARG GIGAAM_MODEL=v3_e2e_rnnt
# Токен Hugging Face передаётся секретом сборки (id=hf_token), а не ARG/ENV:
# он не попадает в метаданные образа (docker history) и не ломает кэш слоёв.
# docker compose передаёт секрет сам (см. секцию secrets в docker-compose.yml).
# Для ручной сборки: docker build --secret id=hf_token,env=HF_TOKEN .
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    GIGAAM_MODEL="$GIGAAM_MODEL" python prepare_models.py

ENV GIGAAM_MODEL=${GIGAAM_MODEL}

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
