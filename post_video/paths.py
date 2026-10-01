"""post-video パス解決ヘルパ。

すべてのパイプラインスクリプトはこれを使って topic 配下の path を解決する。

ディレクトリレイアウト:
  content_pipeline/posts/<topic>/
    slides.json              ← 入力(speech/caption/30 slide、shared_from 参照可)
    slides_raw/<NN>.png      ← gpt-image-2 生成 1536×1024(一部は shared から借りる)
    bgm_config.yaml          ← BGM 3 トラック + section 境界(任意)
    work/
      audio/<NN>_<LL>.wav    ← ElevenLabs 合成出力(shared なら shared から借りる)
      blobs/<sid>/labels.npz ← blob_split
      orders/<sid>_order.json← orderer 出力(resolve 済み)
      anim/<NN>.mp4          ← blob_renderer 出力(1536×1024)
      clips/<NN>.mp4         ← ffmpeg per-slide(1920×1080+音声+字幕)
    final.mp4                ← canonical 最終版(BGM 入り)

shared_slides 共通プール:
  content_pipeline/shared_slides/<shared_id>/
    slide.json               ← lines / title / slide_prompt(既存メタから抽出)
    raw.png                  ← 1536×1024 完成画像
    labels.npz               ← blob_split mask
    order.json               ← reveal アニメ計画
    anim.mp4                 ← blob_renderer 完成 mp4
    line_<LL>.wav            ← ElevenLabs 合成 wav
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return ROOT


def _pipeline_root() -> Path:
    """POST_VIDEO_ROOT env が設定されていればそれを使い、未設定なら従来の ROOT/content_pipeline を返す。"""
    import os
    env = os.environ.get("POST_VIDEO_ROOT", "")
    if env:
        return Path(env).resolve()
    return ROOT / "content_pipeline"


def _posts_subdir() -> str:
    """POST_VIDEO_POSTS_SUBDIR env で切替 (default: posts, 別系統起動時: alt_posts)."""
    import os
    return os.environ.get("POST_VIDEO_POSTS_SUBDIR", "posts").strip() or "posts"


def topic_dir(topic: str) -> Path:
    """topic ディレクトリ。 default: content_pipeline/posts/<topic> / 別系統起動時: content_pipeline/alt_posts/<topic>."""
    return _pipeline_root() / _posts_subdir() / topic


def slides_json(topic: str) -> Path:
    return topic_dir(topic) / "slides.json"


def slides_raw_dir(topic: str) -> Path:
    return topic_dir(topic) / "slides_raw"


def work_dir(topic: str) -> Path:
    return topic_dir(topic) / "work"


def audio_dir(topic: str) -> Path:
    return work_dir(topic) / "audio"


def blobs_dir(topic: str) -> Path:
    return work_dir(topic) / "blobs"


def blobs_overlay_dir(topic: str) -> Path:
    """blob ID overlay PNG 保存先 (mapping強化)。"""
    return work_dir(topic) / "blobs_overlay"


def orders_dir(topic: str) -> Path:
    return work_dir(topic) / "orders"


def anim_dir(topic: str) -> Path:
    return work_dir(topic) / "anim"


def clips_dir(topic: str) -> Path:
    return work_dir(topic) / "clips"


def final_mp4(topic: str, with_bgm: bool = True) -> Path:
    """BGM 入りなら final.mp4(canonical)、なしなら final_no_bgm.mp4。"""
    return topic_dir(topic) / ("final.mp4" if with_bgm else "final_no_bgm.mp4")


def bgm_config(topic: str) -> Path:
    return topic_dir(topic) / "bgm_config.yaml"


# === shared slide pool ===

def shared_slides_root() -> Path:
    return _pipeline_root() / "shared_slides"


def shared_slide_dir(shared_id: str) -> Path:
    return shared_slides_root() / shared_id


def env_file() -> Path:
    return ROOT / ".env"
