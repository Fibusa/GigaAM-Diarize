# syntax=docker/dockerfile:1

# Лёгкий образ веб-интерфейса: без torch/CUDA — только прокси и статика.
# Версии зависимостей совпадают с extra "api" pyproject.toml основного проекта.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ffmpeg нужен для перекодирования загрузок (MP4/M4A/WebM и т.п.) в
# 16 кГц mono WAV перед отправкой в API: у контейнерных форматов с
# edit list сдвигом аудиопотока pyannote не может декодировать окна
# через torchcodec (дефицит сэмплов, задание падает).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Зависимости отдельным слоем: пересобираются только при изменении requirements.txt.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Код веб-интерфейса.
COPY app.py index.html style.css app.js ./

# Внутри compose-сети API доступен по имени сервиса gigaam-api
# (переопределяется переменной окружения при запуске).
ENV API_BASE_URL=http://gigaam-api:8000 \
    WEBUI_PORT=8001

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
