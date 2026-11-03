"""
Предзагрузка всех моделей, необходимых сервису, в локальный кэш.

Запускается на этапе сборки Docker-образа (слой RUN в Dockerfile), чтобы
первый запрос к API не ждал скачивания чекпоинтов и пайплайнов.

Что кэшируется:
  1. Чекпоинт GigaAM (GIGAAM_MODEL, по умолчанию v3_e2e_rnnt) -> ~/.cache/gigaam.
     Качается с CDN, токен не нужен.
  2. pyannote/segmentation-3.0 — модель VAD -> ~/.cache/huggingface.
  3. pyannote/speaker-diarization-3.1 и её подмодели (сегментация, эмбеддинги).

Для пунктов 2-3 нужен HF_TOKEN (переменная окружения или аргумент сборки).
Если токена нет, скрипт НЕ ломает сборку: печатает предупреждение, и модели
скачаются при первом запросе (прежнее поведение).
"""

import os
import sys

import torch

import gigaam
from gigaam.vad_utils import get_diarization_pipeline, get_pipeline


def _step(title: str, func) -> bool:
    """Выполняет один шаг предзагрузки, не прерывая сборку при ошибке."""
    print(f"[prepare_models] {title} ...", flush=True)
    try:
        func()
    except Exception as err:  # noqa: BLE001 - сборка не должна падать из-за кэша
        print(f"[prepare_models] WARNING: {title} failed: {err}", flush=True)
        return False
    print(f"[prepare_models] {title} - OK", flush=True)
    return True


def main() -> int:
    device = torch.device("cpu")
    model_name = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
    ok = True

    ok &= _step(
        f"Loading GigaAM checkpoint '{model_name}'",
        lambda: gigaam.load_model(model_name, device=device),
    )

    ok &= _step(
        "Loading pyannote VAD (segmentation-3.0)",
        lambda: get_pipeline(device),
    )

    ok &= _step(
        "Loading pyannote diarization pipeline (speaker-diarization-3.1)",
        lambda: get_diarization_pipeline(device),
    )

    if not ok:
        print(
            "[prepare_models] WARNING: some models were not cached "
            "(usually HF_TOKEN is missing); they will be downloaded "
            "on the first API request instead.",
            flush=True,
        )
        return 0

    print("[prepare_models] All models are cached.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
