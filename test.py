import gigaam
import os
import time
# Load test audio
audio_path = "tests/audio_examples/audio.wav"
long_audio_path = "tests/audio_examples/audio_long.wav"

# Audio embeddings
model_name = "v3_ssl"       # Options: `v1_ssl`, `v2_ssl`, `v3_ssl`, `multilingual_ssl`, `multilingual_large_ssl`
model = gigaam.load_model(model_name)
embedding, _ = model.embed_audio(audio_path)
print(embedding)

# ASR
model_name = "v3_e2e_rnnt"  # Options: any model version with suffix `_ctc` or `_rnnt`
model = gigaam.load_model(model_name)
transcription = model.transcribe(audio_path)
print(transcription)

# ASR with word-level timestamps
result = model.transcribe(audio_path, word_timestamps=True)
for word in result.words:
    print(f"  [{word.start:.2f} - {word.end:.2f}] {word.text}")

os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
result = model.transcribe_longform(long_audio_path)
for segment in result:
   print(f"[{gigaam.format_time(segment.start)} - {gigaam.format_time(segment.end)}]: {segment.text}")

# ASR + speaker diarization in one call (words and segments get speaker labels)
t0 = time.perf_counter()
merged = model.transcribe_and_diarize(long_audio_path)
print(f"\nTranscription+diarization took {time.perf_counter() - t0:.1f} sec")

speakers = sorted(
    {s.speaker for s in merged if s.speaker}
    | {w.speaker for w in merged.words if w.speaker}
)
print(f"Detected {len(speakers)} speakers: {', '.join(speakers)}")
no_speaker = sum(1 for w in merged.words if w.speaker is None)
print(f"Words: {len(merged.words)} total, {no_speaker} without a confident speaker")
for segment in merged:
    print(
        f"[{gigaam.format_time(segment.start)} -"
        f" {gigaam.format_time(segment.end)}] {segment.speaker}: {segment.text}"
    )
