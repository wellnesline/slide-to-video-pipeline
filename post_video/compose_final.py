"""subtitle-as-artifact パイプの 最終合成ステップ。

build.py --no-subtitle で焼き込まずに作った final mp4 (字幕無し) に、 各 slide の
ASS を global timestamp に offset した「global ASS」 を ffmpeg で 1 回だけ焼き込む。

これにより字幕は build まで遡らず編集できる first-class artifact になり、
UI overlay 編集の WYSIWYG が「動画ピクセルとオーバーレイの divergence」 を
気にしなくて済むようになる。

Usage:
  python -m src.post_video.compose_final --topic <post-id>
  python -m src.post_video.compose_final --topic <post-id> --input custom.mp4 --output custom_subtitled.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import paths
from .slides import load_slides

_ASS_DIALOGUE_RE = re.compile(r"^Dialogue:\s*([^,]*,)([^,]+),([^,]+),(.*)$")
_TIME_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{2})$")


def _parse_ass_time(s: str) -> float:
    m = _TIME_RE.match(s.strip())
    if not m:
        return 0.0
    h, mm, ss, cc = m.groups()
    return int(h) * 3600 + int(mm) * 60 + int(ss) + int(cc) / 100.0


def _fmt_ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:01d}:{m:02d}:{s:05.2f}"


def _video_duration(path: Path) -> float:
    """clip mp4 の video stream の duration (秒) を返す。

    drift bug 修正: container `format.duration = max(audio, video)` を
    使うと ffmpeg subtitles filter が video PTS に焼く性質と矛盾し、 後半で字幕が
    数百 ms 早出しする事故が発生した。 ASS dialogue 時刻は **video
    stream 累積** に乗せるのが正解。

    空ストリーム fallback:
      streams[] 空 / KeyError / IndexError / duration=="N/A" → format.duration で
      再 probe、 stderr に WARN を吐く(sweep バッチが grep する接頭辞固定)。
      再 probe も失敗なら RuntimeError を上流伝播(握り潰さない)。
    """
    r = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration", "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(r.stdout or "{}")
        dur = data["streams"][0]["duration"]
        if dur in (None, "N/A", ""):
            raise ValueError("duration N/A")
        return float(dur)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        # fallback: format.duration で再 probe
        r2 = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-print_format", "json", str(path)],
            capture_output=True, text=True,
        )
        try:
            d = float(json.loads(r2.stdout)["format"]["duration"])
        except Exception as e:
            raise RuntimeError(f"_video_duration: cannot probe {path}: {e}")
        sys.stderr.write(
            f"[compose_final WARN] video stream empty in {path.name}, "
            f"fallback to format.duration ({d:.3f}s)\n"
        )
        return d


def _read_ass_header(per_slide_ass: Path) -> str:
    """ASS の [Script Info] + [V4+ Styles] ヘッダ部分(最初の `[Events]` までを除く)。"""
    text = per_slide_ass.read_text(encoding="utf-8")
    idx = text.find("[Events]")
    if idx < 0:
        raise ValueError(f"[Events] not found in {per_slide_ass}")
    return text[:idx]


def _read_ass_events(per_slide_ass: Path, offset_sec: float) -> list[str]:
    """ASS の Dialogue events を offset_sec 加算して返す。"""
    out: list[str] = []
    in_events = False
    for line in per_slide_ass.read_text(encoding="utf-8").splitlines():
        if line.startswith("[Events]"):
            in_events = True
            continue
        if not in_events:
            continue
        if line.startswith("Format:"):
            continue
        if not line.startswith("Dialogue:"):
            continue
        m = _ASS_DIALOGUE_RE.match(line)
        if not m:
            continue
        prefix, start_str, end_str, rest = m.groups()
        st = _parse_ass_time(start_str) + offset_sec
        et = _parse_ass_time(end_str) + offset_sec
        out.append(f"Dialogue: {prefix}{_fmt_ass_time(st)},{_fmt_ass_time(et)},{rest}")
    return out


def build_global_ass(topic: str) -> Path:
    """全 slide の per-slide ASS を global timestamp に offset してマージ。

    各 clip mp4 の **video stream duration** で offset を確定するので、 ffmpeg
    subtitles filter が video PTS に焼く性質と完全同期する。
    修正: container `format.duration = max(audio, video)` を使ってい
    た旧実装は、 後半 clip で audio が video より長い分だけ字幕が早出しする drift
    を累積させていた(+250ms 級のずれ)。
    """
    slides = load_slides(topic)
    work_d = paths.work_dir(topic)
    out_ass = work_d / "_global.ass"

    # ヘッダ: 1 つの slide ASS から流用 (全 slide 同じ style)
    header_template = None
    events: list[str] = []
    offset = 0.0

    # slide order = build パイプの concat 順 = slides 順 (slide_concat に含まれる id のみ)
    audio_d = paths.audio_dir(topic)
    clips_d = paths.clips_dir(topic)
    for s in slides:
        sid = s["id"]
        clip = clips_d / f"{sid}.mp4"
        concat_wav = audio_d / f"{sid}_concat.wav"
        if not clip.exists():
            print(f"  [global ASS skip] {sid}: clip {clip.name} 不在")
            continue
        per_ass = work_d / f"_tmp_{sid}.ass"
        if not per_ass.exists():
            print(f"  [global ASS skip] {sid}: per-slide ASS 不在")
            offset += _video_duration(clip)
            continue
        if header_template is None:
            header_template = _read_ass_header(per_ass)
        # Dialogue events を offset 加算で global timeline に乗せる
        evts = _read_ass_events(per_ass, offset)
        events.extend(evts)
        # offset 更新は video stream 実 duration(audio stream は若干長いが、
        # 字幕は video PTS に焼かれるため video 基準で累積)
        offset += _video_duration(clip)

    if header_template is None:
        raise RuntimeError("どの slide にも per-slide ASS が無い。 build を先に実行してください。")

    header = header_template
    if not header.endswith("\n"):
        header += "\n"
    body = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        + "\n".join(events) + "\n"
    )
    out_ass.write_text(header + body, encoding="utf-8")
    print(f"[global ASS] {out_ass.name}: {len(events)} events, total {offset:.2f}s")
    return out_ass


def burn_subtitle(input_mp4: Path, ass_path: Path, output_mp4: Path) -> None:
    """ffmpeg subtitles filter で字幕を焼き込む(video のみ再エンコード、 音声 copy)。"""
    ass_escaped = str(ass_path).replace("'", "\\'").replace(":", "\\:")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_mp4),
        "-vf", f"ass='{ass_escaped}'",
        "-c:v", "h264_videotoolbox", "-b:v", "4M", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_mp4),
    ]
    print(f"  [burn] {input_mp4.name} + {ass_path.name} → {output_mp4.name}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError("ffmpeg subtitle burn failed")


def _meta_path(topic: str) -> Path:
    return paths.clips_dir(topic) / "_meta.json"


def _read_meta(topic: str) -> dict:
    """clips/_meta.json を読む。 不在なら legacy 互換 (字幕焼き済 / 旧パイプ) を返す。

    以前の <post-id> は全部 _meta.json 不在の旧パイプ = 字幕焼き済。
    そのため compose_final を不在状態で呼ぶと 2 重焼きになるリスクあり、 デフォルトで防御。
    """
    f = _meta_path(topic)
    if not f.exists():
        return {"subtitle_burned": True, "legacy": True}
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        d.setdefault("subtitle_burned", True)
        return d
    except Exception:
        return {"subtitle_burned": True, "legacy": True, "_parse_error": True}


def _write_meta_update(topic: str, **updates) -> None:
    meta = _read_meta(topic)
    meta.update(updates)
    meta["subtitle_composed_at"] = datetime.now().isoformat(timespec="seconds")
    _meta_path(topic).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--input", default=None,
                    help="字幕を焼く対象 mp4 (default: final_no_bgm.mp4)")
    ap.add_argument("--output", default=None,
                    help="出力 mp4 (default: input と同じ場所 + .subtitled.mp4)")
    ap.add_argument("--in-place", action="store_true",
                    help="output 指定無視で input を直接上書き(中間 .tmp.mp4 経由)")
    ap.add_argument("--force", action="store_true",
                    help="meta が subtitle_burned=True (字幕既焼き) でも実行(2重焼き危険)")
    args = ap.parse_args(argv)

    topic = args.topic
    meta = _read_meta(topic)
    if meta.get("subtitle_burned") and not args.force:
        sys.stderr.write(
            f"[compose_final] clips/_meta.json が subtitle_burned=True を示しています。 "
            f"per-clip 段階で字幕焼き済のため compose_final は不要 (二重焼き防止)。 "
            f"強制実行するなら --force。\n"
        )
        return 1

    print(f"=== compose_final: {topic} ===")
    global_ass = build_global_ass(topic)

    input_mp4 = Path(args.input) if args.input else paths.final_mp4(topic, with_bgm=False)
    if not input_mp4.exists():
        sys.stderr.write(f"input mp4 not found: {input_mp4}\n")
        return 1
    if args.in_place:
        tmp_out = input_mp4.with_suffix(".tmp.mp4")
        try:
            burn_subtitle(input_mp4, global_ass, tmp_out)
            tmp_out.replace(input_mp4)
            final_out = input_mp4
        except Exception:
            # 失敗時は孤児化した .tmp.mp4 を後始末 (次回起動時の混乱防止)
            if tmp_out.exists():
                try: tmp_out.unlink()
                except Exception: pass
            raise
    else:
        final_out = Path(args.output) if args.output else input_mp4.with_suffix(".subtitled.mp4")
        burn_subtitle(input_mp4, global_ass, final_out)
    _write_meta_update(topic, final_subtitled_output=str(final_out.name))
    print(f"\n=== done → {final_out} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
