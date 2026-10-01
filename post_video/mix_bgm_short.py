"""Shorts 用 BGM mix(単一トラック、 全長 loop / fade)。

mix_bgm.py(本編 3 セクション切替)の Shorts 簡略版。

Usage:
  python -m src.post_video.mix_bgm_short --topic <post-id>/shorts/short-A
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import paths
from .slides import load_slides

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_master import master_video_inplace  # noqa: E402

ROOT = paths.repo_root()
BGM_POOL = ROOT / "shared" / "bgm_collection_bright"

DEFAULTS = {
    "track": "intro_track",
    "volume": 0.05,  # 用途に応じて調整 (例: 0.05)
    "fade_in": 0.5,
    "fade_out": 1.0,
}


def _load_cfg(topic: str) -> dict:
    cfg_path = paths.bgm_config(topic)
    cfg = dict(DEFAULTS)
    if cfg_path.exists():
        try:
            import yaml
            cfg.update(yaml.safe_load(cfg_path.read_text()) or {})
        except ImportError:
            pass
    # slides.json bgm に main があれば使う
    top = json.loads(paths.slides_json(topic).read_text(encoding="utf-8"))
    if "bgm" in top and "main" in top["bgm"]:
        cfg["track"] = top["bgm"]["main"]
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    args = ap.parse_args(argv)

    topic = args.topic
    cfg = _load_cfg(topic)
    voice_mp4 = paths.final_mp4(topic, with_bgm=False)
    out_mp4 = paths.final_mp4(topic, with_bgm=True)
    if not voice_mp4.exists():
        sys.stderr.write(f"missing: {voice_mp4}\n"); return 1

    bgm = BGM_POOL / f"{cfg['track']}.mp3"
    if not bgm.exists():
        sys.stderr.write(f"missing BGM: {bgm}\n"); return 1

    # voice mp4 の duration を取得
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", str(voice_mp4)],
        capture_output=True, text=True,
    )
    total = float(json.loads(r.stdout)["format"]["duration"])
    volume = float(cfg["volume"])
    fade_in = float(cfg["fade_in"])
    fade_out = float(cfg["fade_out"])

    fc = (
        f"[1:a]aloop=loop=-1:size=2147483647,atrim=duration={total},"
        f"asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d={fade_in},"
        f"afade=t=out:st={total - fade_out}:d={fade_out},"
        f"volume={volume}[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(voice_mp4),
        "-i", str(bgm),
        "-filter_complex", fc,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_mp4),
    ]
    print(f"=== BGM mix Shorts (topic={topic}) ===")
    print(f"  voice  : {voice_mp4.name} ({total:.1f}s)")
    print(f"  bgm    : {bgm.name}")
    print(f"  volume : {volume}")
    print(f"  fade   : in={fade_in}s, out={fade_out}s")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:]); return 1
    print(f"=> {out_mp4}")
    # 恒久対応: -16 LUFS マスタリング を <post-id> Shorts にも適用
    master_video_inplace(out_mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
