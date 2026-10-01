"""<post-id>/shorts/short-X 用の縦型 1080×1920 build スクリプト。

build.py(本編横型 1920×1080)の縦型版。 主要差分:
  - 解像度: 1080×1920(縦型 9:16)
  - スライド領域: 中央 1080×1440(上下中央寄せ、 上下黒帯あり)
  - 字幕領域: 下 480px(画面下部、 大きめ font)
  - max_line=18(横幅狭い)、 max_lines=2、 font 72px
  - 横型 raw png(1536×1024)も縦型 raw(1024×1536)も decrease+pad で自動 fit
  - per-slide clip 並列(ProcessPool)

build.py のロジックを最大限再利用(import で共有)。

Usage:
  python -m src.post_video.build_short --topic <post-id>/shorts/short-A
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import paths
from .slides import load_slides
from .build import (
    concat_slide_audio, _sanitize_segments,
    PARTICLES, PARTICLE_COMPOUND, _split_long_chunk,
)

# === 縦型 1080×1920 設定 ===
VIDEO_W, VIDEO_H = 1080, 1920
SLIDE_W = 1080
SLIDE_H = 1440  # スライド領域(上 + 下黒帯あり、 中央寄せ)
# 字幕帯の位置(確定: 横型 raw 1536×1024 が 1080×720 にフィットするため
# 画像下端 ≈ 1080。 字幕は画像直下に置くと自然 + 下部 ~120px の YouTube UI と被らない)
SUB_BAND_TOP = 1080     # 横型画像の下端
SUB_BAND_BOTTOM = 1280  # 字幕中心 ≈ 1180、 画像直下

# 字幕(縦型用に圧縮)
WRAP_STYLE = 2
FONT_NAME = os.environ.get("SUBTITLE_FONT_NAME", "sans-serif")  # ASS 字幕のフォント名(環境に入っているフォントを指定)
FONT_SIZE = 72   # 縦型は font 大きめ
OUTLINE = 5
MAX_LINE = 18    # 縦型は幅狭、 18 字
MAX_LINES = 2
MIN_DUR = 0.6


def split_caption_for_display_short(caption: str, max_line: int = MAX_LINE, max_lines: int = MAX_LINES) -> list[str]:
    """縦型 Shorts 用 caption 分割(max_line=18 で詰める、 短文 entry 防止 chunk 段階吸収)。"""
    s = caption.strip()
    if not s:
        return []
    chunks = re.split(r"(?<=[、。!?!?・」』])", s)
    chunks = [c.strip() for c in chunks if c.strip()] or [s]
    # 短い chunk を前 chunk に吸収(「す。」断片対策)
    MIN_CHUNK = 4
    merged_chunks: list[str] = []
    for c in chunks:
        if merged_chunks and len(c) < MIN_CHUNK and len(merged_chunks[-1]) + len(c) <= max_line * max_lines:
            merged_chunks[-1] += c
        else:
            merged_chunks.append(c)
    chunks = merged_chunks
    refined: list[str] = []
    for c in chunks:
        refined.extend(_split_long_chunk(c, max_line))
    lines: list[str] = []
    cur = ""
    for c in refined:
        if not cur:
            cur = c; continue
        if len(cur) + len(c) <= max_line:
            cur += c
        else:
            lines.append(cur); cur = c
    if cur:
        lines.append(cur)
    entries: list[str] = []
    for i in range(0, len(lines), max_lines):
        entries.append("\\N".join(lines[i:i+max_lines]))
    return entries


def write_ass_short(segments, ass_path: Path, play_w=VIDEO_W, play_h=VIDEO_H,
                    fontname=FONT_NAME, fontsize=FONT_SIZE, outline=OUTLINE,
                    band_top=SUB_BAND_TOP, band_bottom=SUB_BAND_BOTTOM):
    line_h = int(fontsize * 1.18)
    band_mid = (band_top + band_bottom) // 2

    def fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = t - h*3600 - m*60
        return f"{h:01d}:{m:02d}:{s:05.2f}"

    style_mv = play_h - band_mid - line_h // 2
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\nPlayResY: {play_h}\n"
        "ScaledBorderAndShadow: no\n"
        f"WrapStyle: {WRAP_STYLE}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fontname},{fontsize},&H00FFFFFF,&H00000000,1,{outline},0,2,30,30,{style_mv},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    sanitized = _sanitize_segments(segments, min_dur=MIN_DUR)
    body_lines = []
    for st, et, txt in sanitized:
        n_line = txt.count("\\N") + 1
        margin_v = play_h - band_mid - line_h * n_line // 2
        body_lines.append(f"Dialogue: 0,{fmt(st)},{fmt(et)},Default,,0,0,{margin_v},,{txt}")
    ass_path.write_text(header + "\n".join(body_lines) + "\n", encoding="utf-8")


# === per-slide clip(縦型)===

def build_clip_short(topic: str, slide: dict, audio_path: Path) -> Path:
    sid = slide["id"]
    clip_out = paths.clips_dir(topic) / f"{sid}.mp4"
    anim_mp4 = paths.anim_dir(topic) / f"{sid}.mp4"
    raw_fallback = slide["raw_path"]

    # 字幕 segments(複数 entry は force_align で word timestamps 取得して字幕同期する)
    # 元実装は比例分割で粗い → ユーザー視聴で「字幕と音声がズレる」 と指摘
    # 本編 build.py と同じ logic に統一(Shorts で発覚、 fix)
    from . import force_align
    segments: list = []
    for li, line in enumerate(slide["lines"]):
        cap = line["caption"]
        entries = split_caption_for_display_short(cap)
        if not entries:
            continue
        if len(entries) == 1:
            segments.append((line["start"], line["end"], entries[0]))
        else:
            line_wav = slide["audio_paths"][li] if li < len(slide["audio_paths"]) else None
            words: list = []
            if line_wav and line_wav.exists():
                try:
                    words = force_align.transcribe(
                        line_wav, language="ja", initial_prompt=cap,
                    )
                except Exception as e:
                    print(f"  [align WARN] {sid}_{li:02d}: {e}")
            line_dur = line["end"] - line["start"]
            try:
                timings = force_align.align_entries_to_wav(
                    entries, words,
                    line_start_offset=line["start"],
                    line_duration=line_dur,
                )
            except Exception:
                chunk = line_dur / len(entries)
                timings = [(line["start"] + i*chunk, line["start"] + (i+1)*chunk) for i in range(len(entries))]
            for ent, (st, et) in zip(entries, timings):
                segments.append((st, et, ent))
    if segments:
        duration_for_extend = max(line["end"] for line in slide["lines"]) + 0.7
        segs_sorted = sorted(segments, key=lambda s: s[0])
        last_st, last_et, last_txt = segs_sorted[-1]
        segs_sorted[-1] = (last_st, max(last_et, duration_for_extend), last_txt)
        segments = segs_sorted

    ass_path = paths.work_dir(topic) / f"_tmp_{sid}.ass"
    write_ass_short(segments, ass_path)
    ass_escaped = str(ass_path).replace("'", "\\'").replace(":", "\\:")

    duration = max(line["end"] for line in slide["lines"]) + 0.7

    # filter:slide 領域(1080×1440)に decrease+pad、 字幕は ASS で下部
    # 縦型 raw(1024×1536)も横型 raw(1536×1024)も decrease で自動 fit、 余白白
    if anim_mp4.exists():
        filter_complex = (
            f"color=c=white:s={VIDEO_W}x{VIDEO_H}:d={duration}[bg];"
            f"[0:v]scale={SLIDE_W}:{SLIDE_H}:force_original_aspect_ratio=decrease,"
            f"pad={SLIDE_W}:{SLIDE_H}:(ow-iw)/2:(oh-ih)/2:white[s];"
            f"[bg][s]overlay=0:0[withslide];"
            f"[withslide]ass='{ass_escaped}'[v]"
        )
        video_input = ["-i", str(anim_mp4)]
    else:
        filter_complex = (
            f"color=c=white:s={VIDEO_W}x{VIDEO_H}:d={duration}[bg];"
            f"[0:v]scale={SLIDE_W}:{SLIDE_H}:force_original_aspect_ratio=decrease,"
            f"pad={SLIDE_W}:{SLIDE_H}:(ow-iw)/2:(oh-ih)/2:white[s];"
            f"[bg][s]overlay=0:0[withslide];"
            f"[withslide]ass='{ass_escaped}'[v]"
        )
        video_input = ["-loop", "1", "-i", str(raw_fallback)]

    cmd = [
        "ffmpeg", "-y", *video_input, "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "1:a",
        "-c:v", "h264_videotoolbox", "-b:v", "4M", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration}", "-shortest",
        str(clip_out),
    ]
    print(f"  [C] {sid} compose ({duration:.2f}s, {len(segments)} subtitles)")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError(f"ffmpeg failed for {sid}")
    return clip_out


def concat_clips(topic: str, clips: list[Path]) -> Path:
    listfile = paths.work_dir(topic) / "_concat.txt"
    listfile.write_text("\n".join(f"file '{p}'" for p in clips))
    out = paths.final_mp4(topic, with_bgm=False)
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)]
    print(f"[D] concat → {out}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError("concat failed")
    return out


def _build_clip_short_worker(args: tuple) -> tuple[str, Path | None, str]:
    topic, slide, audio_path = args
    try:
        clip = build_clip_short(topic, slide, audio_path)
        return (slide["id"], clip, "OK")
    except Exception as e:
        return (slide["id"], None, f"{type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-qa", action="store_true",
                    help="Step 0 の qa_post_video pre-build BLOCK gate を skip (debug 用、 通常使うな)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Step 0.5 の verify_synth Whisper gate を skip (debug 用)")
    args = ap.parse_args(argv)

    topic = args.topic
    repo_root = Path(__file__).resolve().parents[2]

    # Step 0: 構造 QA (caption/speech 乖離 BLOCK gate)
    # 監査 3 結論: Shorts は scripter agent スコープ外で
    # 自己申告 lint が機能せず γ パターン乖離が連発したため、 build パイプ入口でハードガード。
    if not args.skip_qa:
        print(f"\n{'=' * 60}\n=== Step 0: qa_post_video (pre-build BLOCK gate) ===\n{'=' * 60}", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "src.post_video.qa_post_video", "--topic", topic],
            cwd=str(repo_root),
        )
        # exit 1 = BLOCK あり、 exit 2 = WARN のみ (build 続行 OK)
        if r.returncode == 1:
            sys.stderr.write(f"\n[build_short] qa_post_video BLOCK at topic={topic}, abort.\n")
            return r.returncode

    # Step 0.5: verify_synth (Whisper で実音声 vs speech 比較、 wav 揃ってる前提)
    # 監査 1 結論 (過去の事故由来): 致命的 mismatch (ratio < 0.5) は build パイプで abort。
    if not args.skip_verify:
        print(f"\n{'=' * 60}\n=== Step 0.5: verify_synth (Whisper TTS sanity gate) ===\n{'=' * 60}", flush=True)
        r = subprocess.run(
            [sys.executable, "-m", "src.post_video.verify_synth",
             "--topic", topic, "--strict"],
            cwd=str(repo_root),
        )
        if r.returncode == 1:
            sys.stderr.write(f"\n[build_short] verify_synth BLOCK at topic={topic}, abort. kana_dict 整備 or speech 修正後に再実行。\n")
            return r.returncode

    paths.audio_dir(topic).mkdir(parents=True, exist_ok=True)
    paths.clips_dir(topic).mkdir(parents=True, exist_ok=True)

    slides = load_slides(topic)

    print("=== Phase A2: 各 slide 内の wav 結合 ===")
    slide_concat: dict[str, Path] = {}
    for s in slides:
        sid = s["id"]
        if not all(p.exists() for p in s["audio_paths"]):
            print(f"  [A2 skip] {sid}: 一部 wav 不在")
            continue
        slide_concat[sid] = concat_slide_audio(paths.audio_dir(topic), sid, s["audio_paths"], s["lines"])

    print(f"\n=== Phase C: 縦型 per-slide clip(workers={args.workers}) ===")
    work_args = [(topic, s, slide_concat[s["id"]])
                 for s in slides if s["id"] in slide_concat]
    clip_paths_dict: dict[str, Path] = {}
    n_fail = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_build_clip_short_worker, a): a[1]["id"] for a in work_args}
        for f in as_completed(futures):
            sid, clip, msg = f.result()
            if clip is not None:
                clip_paths_dict[sid] = clip
                print(f"  [C ok]    {sid}")
            else:
                print(f"  [C error] {sid}: {msg}")
                n_fail += 1
    if n_fail > 0:
        return 1
    clip_paths = [clip_paths_dict[s["id"]] for s in slides if s["id"] in clip_paths_dict]

    print("\n=== Phase D: concat ===")
    concat_clips(topic, clip_paths)
    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
