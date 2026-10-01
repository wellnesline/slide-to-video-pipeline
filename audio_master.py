"""synthetic stub: loudness mastering helper (placeholder, not the original).

Provides the names the pipeline imports. ``master_video_inplace`` here is a no-op that
leaves the file untouched; replace with a real two-pass loudnorm implementation.
"""
from __future__ import annotations
from pathlib import Path

ALIMITER_SUFFIX = ""  # synthetic stub: no limiter filter appended


def master_video_inplace(path: str | Path, target_lufs: float = -16.0) -> None:
    """No-op placeholder. Implement loudness normalization (e.g. ffmpeg loudnorm) here."""
    return None


def pre_bake_loudnorm(src: str | Path, dst: str | Path, target_lufs: float = -16.0) -> None:
    """Placeholder: copies bytes unchanged."""
    Path(dst).write_bytes(Path(src).read_bytes())
