"""slides.json 読込み + shared_from 解決ヘルパ。

slides.json schema:
{
  "title": "post タイトル",
  "voice": "<VOICE_ID>",                // 任意、デフォルトは synth.py の VOICE_DEFAULT
  "bgm": {                              // 任意、なければ デフォルト
    "intro": "intro_track",
    "main":  "main_track",
    "outro": "outro_track"
  },
  "slides": [
    {"shared_from": "intro_shared"},
    {"id": "02", "title": "...", "lines": [{"speech": "...", "caption": "..."}, ...]},
    ...
    {"shared_from": "outro_shared_a"},
    {"shared_from": "outro_shared_b"}
  ]
}

shared_from を解決すると、各 slide は以下に正規化される:
{
  "id":          "01"..."NN",          # 自動連番(slides 配列順)
  "title":       "...",
  "lines":       [{"speech", "caption"}, ...],
  "shared_id":   "intro_shared" or None,
  "raw_path":    Path,                  # 完成画像 1536x1024
  "labels_path": Path,                  # blob_split labels.npz
  "order_path":  Path,                  # order JSON(resolve 済み)
  "anim_path":   Path or None,          # blob_renderer mp4
  "audio_paths": [Path, ...]            # 各 line の wav(post 個別なら未生成 path、shared なら既存)
}
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from . import paths


def _shared_meta(shared_id: str) -> dict[str, Any]:
    p = paths.shared_slide_dir(shared_id) / "slide.json"
    if not p.exists():
        raise FileNotFoundError(f"shared slide が見つかりません: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_slides(topic: str) -> list[dict[str, Any]]:
    """slides.json を読み込み、shared_from を解決して正規化済 slide 配列を返す。

    slides 配列内の文字列(例: "_SECTION_INTRO": "...")はコメントとして無視。
    """
    raw = json.loads(paths.slides_json(topic).read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    real_idx = 0
    for entry in raw.get("slides", []):
        if not isinstance(entry, dict):
            continue  # コメント文字列(例: "_SECTION_INTRO: ...")はスキップ
        real_idx += 1
        sid = f"{real_idx:02d}"
        if "shared_from" in entry:
            shared_id = entry["shared_from"]
            meta = _shared_meta(shared_id)
            sdir = paths.shared_slide_dir(shared_id)
            audio_paths = [
                sdir / f"line_{li:02d}.wav"
                for li in range(len(meta["lines"]))
            ]
            slide = {
                "id":          sid,
                "title":       meta.get("title", ""),
                "lines":       meta["lines"],
                "shared_id":   shared_id,
                "raw_path":    sdir / "raw.png",
                "labels_path": sdir / "labels.npz",
                "order_path":  sdir / "order.json",
                "anim_path":   sdir / "anim.mp4",
                "audio_paths": audio_paths,
                "slide_prompt": meta.get("slide_prompt", ""),
            }
        else:
            # post 個別 slide
            audio_paths = [
                paths.audio_dir(topic) / f"{sid}_{li:02d}.wav"
                for li in range(len(entry["lines"]))
            ]
            slide = {
                "id":          sid,
                "title":       entry.get("title", ""),
                "lines":       entry["lines"],
                "shared_id":   None,
                "raw_path":    paths.slides_raw_dir(topic) / f"{sid}.png",
                "labels_path": paths.blobs_dir(topic) / f"slide_{sid}" / "labels.npz",
                "order_path":  paths.orders_dir(topic) / f"slide_{sid}_order.json",
                "anim_path":   paths.anim_dir(topic) / f"{sid}.mp4",
                "audio_paths": audio_paths,
                "slide_prompt": entry.get("slide_prompt", ""),
            }
        out.append(slide)
    return out


def load_top(topic: str) -> dict[str, Any]:
    """slides.json の top-level fields(title, voice, bgm 等)を返す。"""
    return json.loads(paths.slides_json(topic).read_text(encoding="utf-8"))
