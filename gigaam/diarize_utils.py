"""Merging of ASR transcription results with PyAnnote speaker diarization."""

from typing import TYPE_CHECKING, List, Optional, Tuple

from .types import LongformTranscriptionResult, Segment, Word

if TYPE_CHECKING:
    from pyannote.core import Annotation


def _best_speaker(
    start: float,
    end: float,
    turns: List[Tuple[float, float, str]],
    min_overlap: float,
) -> Optional[str]:
    """
    Return the speaker whose turn overlaps [start, end] the most.
    The label is returned only when the best turn covers at least `min_overlap`
    of the interval duration; when two turns cover the interval equally well,
    the shorter (more specific) turn wins, which attributes words from an
    overlapping turn to the interrupting speaker instead of the long
    background turn.
    """
    duration = end - start
    best_label: Optional[str] = None
    best_fraction = 0.0
    best_turn_duration = float("inf")
    for turn_start, turn_end, label in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap <= 0.0:
            continue
        fraction = 1.0 if duration <= 0.0 else overlap / duration
        turn_duration = turn_end - turn_start
        if fraction > best_fraction or (
            fraction == best_fraction and turn_duration < best_turn_duration
        ):
            best_fraction = fraction
            best_label = label
            best_turn_duration = turn_duration
    if best_label is None or best_fraction < min_overlap:
        return None
    return best_label


def assign_speakers(
    result: LongformTranscriptionResult,
    annotation: "Annotation",
    min_overlap: float = 0.3,
) -> LongformTranscriptionResult:
    """
    Assign speaker labels to a transcription using diarization turns.
    Every word gets the label of the speaker whose turn overlaps it the most
    (at least `min_overlap` of the word duration), otherwise no label. Then
    segments are split at speaker changes, so every resulting segment contains
    words of a single speaker; words without a confident label join the
    surrounding speaker run. Segments without word timestamps fall back to
    assigning a single speaker to the whole segment. The input is not modified.
    """
    turns = [
        (turn.start, turn.end, str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    segments: List[Segment] = []
    for seg in result.segments:
        if not seg.words:
            segments.append(
                Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    words=seg.words,
                    speaker=_best_speaker(seg.start, seg.end, turns, min_overlap),
                )
            )
            continue
        words = [
            Word(
                text=w.text,
                start=w.start,
                end=w.end,
                speaker=_best_speaker(w.start, w.end, turns, min_overlap),
            )
            for w in seg.words
        ]
        segments.extend(_split_segment(seg, words))
    return LongformTranscriptionResult(segments=segments)


def _split_segment(seg: Segment, words: List[Word]) -> List[Segment]:
    """
    Group words into speaker runs and build one segment per run.
    Words without a confident label follow the previous confident word; leading
    unassigned words are attached to the first confident speaker that follows.
    """
    filled: List[Optional[str]] = [w.speaker for w in words]
    previous: Optional[str] = None
    for i, speaker in enumerate(filled):
        if speaker is None:
            filled[i] = previous
        else:
            previous = speaker
    following: Optional[str] = None
    for i in reversed(range(len(filled))):
        if filled[i] is None:
            filled[i] = following
        else:
            following = filled[i]

    runs: List[List[int]] = []
    current: List[int] = []
    for i, speaker in enumerate(filled):
        if current and filled[current[-1]] != speaker:
            runs.append(current)
            current = []
        current.append(i)
    if current:
        runs.append(current)

    if len(runs) == 1:
        return [
            Segment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                words=words,
                speaker=filled[0],
            )
        ]
    return [
        Segment(
            text=" ".join(words[i].text for i in run),
            start=words[run[0]].start,
            end=words[run[-1]].end,
            words=[words[i] for i in run],
            speaker=filled[run[0]],
        )
        for run in runs
    ]
