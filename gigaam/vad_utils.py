import json
import os
import warnings
from typing import List, Optional, Tuple, cast

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pyannote.audio import Model, Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Annotation
from torch.torch_version import TorchVersion

from .preprocess import load_audio

_PIPELINE = None

_DIARIZATION_PIPELINE = None

# Community-1 is the current flagship diarization pipeline of pyannote.audio 4.x
# (powerset segmentation + VBx clustering, PLDA re-scoring). It is a gated model:
# its CC-BY-4.0 conditions must be accepted on the Hugging Face model page for
# the HF_TOKEN user. Fallback keeps the service usable with the older 3.1 model.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
FALLBACK_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def resolve_local_model_path(model_id: str) -> str:
    """
    Finds the local path to the model snapshot.
    If it doesn’t exist, it first creates (downloads) the snapshot.
    """
    try:
        return snapshot_download(
            repo_id=model_id,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pass

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            f"Model {model_id} was not found locally, "
            f"and no HF_TOKEN was provided to download it."
        )

    return snapshot_download(
        repo_id=model_id,
        token=hf_token,
    )


def load_segmentation_model(model_id: str) -> Model:
    """
    Loads the segmentation model from a local snapshot.
    If it doesn’t exist, it first creates (downloads) the snapshot.
    """
    local_path = resolve_local_model_path(model_id=model_id)

    with torch.serialization.safe_globals(
        [
            TorchVersion,
            Problem,
            Specifications,
            Resolution,
        ]
    ):
        return Model.from_pretrained(local_path)


def get_pipeline(
    device: torch.device, model_id: str = "pyannote/segmentation-3.0"
) -> Pipeline:
    """
    Retrieves a PyAnnote voice activity detection pipeline and moves it to the specified device.
    The pipeline is loaded only once and reused across subsequent calls.
    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE.to(device)

    model = load_segmentation_model(model_id=model_id)

    _PIPELINE = VoiceActivityDetection(segmentation=model)
    _PIPELINE.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    return _PIPELINE.to(device)


def get_diarization_pipeline(
    device: torch.device, model_id: Optional[str] = None
) -> Pipeline:
    """
    Retrieves a PyAnnote speaker diarization pipeline and moves it to the specified device.
    The pipeline is loaded only once and reused across subsequent calls.
    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.

    By default uses the newer pyannote/speaker-diarization-community-1 pipeline
    (pyannote.audio >= 4.0: powerset segmentation, VBx clustering, PLDA). The model
    can be overridden with the PYANNOTE_DIARIZATION_MODEL environment variable.
    community-1 is gated, so its licence conditions must be accepted on the
    Hugging Face model page for the HF_TOKEN user. If the pipeline cannot be loaded
    (conditions not accepted, no token or model not cached), it falls back to
    pyannote/speaker-diarization-3.1 with a warning.

    Optional hyper-parameter overrides can be supplied as a JSON partial parameter
    dict in PYANNOTE_DIARIZATION_PARAMS, applied via pipeline.instantiate(). Only
    parameters that exist in the pipeline are allowed; for community-1 these are
    e.g. {"segmentation": {"min_duration_off": 0.0}} and
    {"clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}} (VBx clustering —
    do not mix with the 3.1 agglomerative clustering parameters such as
    min_cluster_size).
    """
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is not None:
        return _DIARIZATION_PIPELINE.to(device)

    # An empty env value is treated as "not set" (compose may pass "").
    env_model = os.getenv("PYANNOTE_DIARIZATION_MODEL") or DEFAULT_DIARIZATION_MODEL
    model_id = model_id or env_model
    hf_token = os.getenv("HF_TOKEN")

    try:
        local_path = resolve_local_model_path(model_id=model_id)

        # Sub-models referenced by the pipeline config (segmentation, embedding,
        # plda) are resolved and downloaded by pyannote itself using the token.
        pipeline = Pipeline.from_pretrained(local_path, token=hf_token)

        params_json = os.getenv("PYANNOTE_DIARIZATION_PARAMS")
        if params_json:
            pipeline.instantiate(json.loads(params_json))
    except Exception as exc:  # degrade gracefully: keep the service usable
        if model_id == FALLBACK_DIARIZATION_MODEL:
            raise
        warnings.warn(
            f"Failed to load diarization pipeline '{model_id}' ({exc}); "
            f"falling back to '{FALLBACK_DIARIZATION_MODEL}'. To use "
            f"'{DEFAULT_DIARIZATION_MODEL}', accept its conditions on "
            "https://huggingface.co/pyannote/speaker-diarization-community-1 "
            "for the HF_TOKEN user.",
            stacklevel=2,
        )
        return get_diarization_pipeline(device, model_id=FALLBACK_DIARIZATION_MODEL)

    _DIARIZATION_PIPELINE = pipeline
    return _DIARIZATION_PIPELINE.to(device)


def segment_audio_file(
    wav_file: str,
    sr: int,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
    """
    Segments an audio waveform into smaller chunks based on speech activity.
    The segmentation is performed using a PyAnnote voice activity detection pipeline.
    """

    audio = load_audio(wav_file)
    pipeline = get_pipeline(device)
    # Feed the already decoded 16 kHz mono waveform instead of the file path:
    # avoids a second decode inside pyannote and keeps VAD timings aligned with
    # the exact samples sliced below.
    sad_segments = cast(
        Annotation,
        pipeline({"waveform": audio.unsqueeze(0), "sample_rate": sr}),
    )

    segments: List[torch.Tensor] = []
    curr_duration = 0.0
    curr_start = 0.0
    curr_end = 0.0
    boundaries: List[Tuple[float, float]] = []

    def _update_segments(curr_start: float, curr_end: float, curr_duration: float):
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    # Concat segments from pipeline into chunks for asr according to max/min duration
    # Segments longer than strict_limit_duration are split manually
    for segment in sad_segments.get_timeline().support():
        start = max(0, segment.start)
        end = min(audio.shape[0] / sr, segment.end)
        if curr_duration == 0.0:
            curr_start = start
        elif curr_duration > new_chunk_threshold and (
            curr_duration + (end - curr_end) > max_duration
            or curr_duration > min_duration
        ):
            _update_segments(curr_start, curr_end, curr_duration)
            curr_start = start
        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _update_segments(curr_start, curr_end, curr_duration)

    return segments, boundaries
