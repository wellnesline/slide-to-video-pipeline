"""post-video — 台本(slides.json)から動画を組み立てる共通モジュール。

トピックごと(<topic>/)の動画ビルドを汎用化したもの。

エントリポイント:
  - python -m post_video.synth          --topic <post-id>
  - python -m post_video.parallel_render --topic <post-id> [--workers 6]
  - python -m post_video.build          --topic <post-id>
  - python -m post_video.mix_bgm        --topic <post-id>

shared slide(挨拶/締め等の固定セリフ)は shared_slides/ にプール。
slides.json で `{"shared_from": "intro_shared"}` と指定すれば assets を流用。
"""
from . import paths

__all__ = ["paths"]