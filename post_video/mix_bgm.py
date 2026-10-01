"""<post-id> の BGM 3 セクション切替 mix。

bgm_config.yaml(任意):
  intro: intro_track          # bgm_collection_bright/<name>.mp3
  main:  main_track
  outro: outro_track
  volume: 0.05                # 例: 0.05
  intro_slides: 3             # 先頭 N slide が intro 区間
  outro_slides: 5             # 末尾 N slide が outro 区間
  xfade: 1.0

なければ DEFAULTS を使う。

Usage:
  python -m post_video.mix_bgm --topic <post-id>
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
    "intro": "intro_track",
    "main":  "main_track",
    "outro": "outro_track",
    "volume": 0.05,  # 用途に応じて調整 (例: 0.05)
    "intro_slides": 3,
    "outro_slides": 5,
    "xfade": 1.0,
}


def _load_bgm_cfg(topic: str) -> dict:
    cfg_path = paths.bgm_config(topic)
    cfg = dict(DEFAULTS)
    if cfg_path.exists():
        try:
            import yaml
            cfg.update(yaml.safe_load(cfg_path.read_text()) or {})
        except ImportError:
            print("[mix_bgm] yaml 未 install、defaults 使用")
    return cfg


def _slide_durs(topic: str) -> list[float]:
    """各 slide の duration を clip mp4 から ffprobe で取得。"""
    slides = load_slides(topic)
    out: list[float] = []
    for s in slides:
        clip = paths.clips_dir(topic) / f"{s['id']}.mp4"
        if not clip.exists():
            out.append(0.0)
            continue
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_format", "-print_format", "json", str(clip)],
            capture_output=True, text=True,
        )
        out.append(float(json.loads(r.stdout)["format"]["duration"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    args = ap.parse_args(argv)

    topic = args.topic
    cfg = _load_bgm_cfg(topic)
    voice_mp4 = paths.final_mp4(topic, with_bgm=False)
    out_mp4 = paths.final_mp4(topic, with_bgm=True)
    if not voice_mp4.exists():
        sys.stderr.write(f"missing: {voice_mp4}\n"); return 1

    bgm1 = BGM_POOL / f"{cfg['intro']}.mp3"
    bgm2 = BGM_POOL / f"{cfg['main']}.mp3"
    bgm3 = BGM_POOL / f"{cfg['outro']}.mp3"
    for p in (bgm1, bgm2, bgm3):
        if not p.exists():
            sys.stderr.write(f"missing BGM: {p}\n"); return 1

    durs = _slide_durs(topic)
    n = len(durs)
    intro_n = int(cfg["intro_slides"])
    outro_n = int(cfg["outro_slides"])

    # 小規模 slide(intro_n + outro_n が n を超える場合)の fallback:
    # 比例的に縮める。最低 1 slide ずつは確保。
    if intro_n + outro_n >= n:
        # n=4, intro=3, outro=5 → ぶつかるので圧縮
        # 単純戦略: intro=ceil(n/3), outro=ceil(n/3)、main は中央 1 slide以上残す
        if n >= 3:
            intro_n = max(1, n // 3)
            outro_n = max(1, n // 3)
        else:
            # n=1 or 2 の極端ケース: BGM 切替なしで 1 トラックで通す方が無難だが、
            # 暫定で intro=1, outro=0, main=残り に
            intro_n = 1
            outro_n = 0
        print(f"[mix_bgm] section size auto-adjusted: intro={intro_n}, outro={outro_n} (n_slides={n})")

    sec1 = sum(durs[:intro_n])
    sec3 = sum(durs[n - outro_n:]) if outro_n > 0 else 0.0
    sec2 = sum(durs[intro_n:n - outro_n]) if (intro_n + outro_n) < n else 0.0
    total = sec1 + sec2 + sec3
    xfade = float(cfg["xfade"])
    volume = float(cfg["volume"])

    trim1 = sec1 + xfade
    trim2 = sec2 + xfade
    trim3 = sec3 + xfade

    fc = (
        f"[1:a]atrim=duration={trim1},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.5[b1];"
        f"[2:a]aloop=loop=-1:size=2147483647,atrim=duration={trim2},asetpts=PTS-STARTPTS[b2];"
        f"[3:a]atrim=duration={trim3},asetpts=PTS-STARTPTS,afade=t=out:st={sec3 - 0.5}:d=1.5[b3];"
        f"[b1][b2]acrossfade=d={xfade}[b12];"
        f"[b12][b3]acrossfade=d={xfade}[bgm_full];"
        f"[bgm_full]atrim=duration={total},volume={volume}[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(voice_mp4),
        "-i", str(bgm1), "-i", str(bgm2), "-i", str(bgm3),
        "-filter_complex", fc,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        # 追加: ブラウザ progressive playback のため moov atom を先頭移動
        # (Faststart 化、 別の再生環境 で video tag 再生時 数秒で停止する事象対策)
        "-movflags", "+faststart",
        "-shortest", str(out_mp4),
    ]
    print(f"=== BGM mix (topic={topic}) ===")
    print(f"  voice  : {voice_mp4.name}")
    print(f"  intro  : {bgm1.name}  ({sec1:.1f}s, {intro_n} slide)")
    print(f"  main   : {bgm2.name}  ({sec2:.1f}s, loop)")
    print(f"  outro  : {bgm3.name}  ({sec3:.1f}s, {outro_n} slide)")
    print(f"  volume : {volume}")
    print(f"  xfade  : {xfade}s")
    print(f"  total  : {total:.1f}s")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:]); return 1
    print(f"=> {out_mp4}")
    # 恒久対応: -16 LUFS マスタリング を <post-id> にも適用
    master_video_inplace(out_mp4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
