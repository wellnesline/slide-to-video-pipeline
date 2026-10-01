"""<post-id> の build パイプライン。

入力前提:
  - synth.py 完了(work/audio/<sid>_<li>.wav 揃い)
  - parallel_render.py 完了(work/anim/<sid>.mp4 揃い)

出力:
  - work/clips/<sid>.mp4(ffmpeg per-slide 1920x1080 + 字幕 + 音声)
  - final_no_bgm.mp4(全 clip concat)

Usage:
  python -m post_video.build --topic <post-id>
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import paths
from .slides import load_slides

VIDEO_W, VIDEO_H = 1920, 1080
SLIDE_H = 880

# ASS 字幕パラメータ
WRAP_STYLE = 2
FONT_NAME = os.environ.get("SUBTITLE_FONT_NAME", "sans-serif")  # ASS 字幕のフォント名(環境に入っているフォントを指定)
FONT_SIZE = 80
OUTLINE = 4
MAX_LINE = 28
MAX_LINES = 2
MIN_DUR = 0.6
SENTENCE_END = "。!?!?"  # 句点扱い、 line packing で強制行切替 + 出口検証で位置チェック


# === Audio concat (Phase A2) ===

# 各 line 音声の 文頭 頭切れ 対策(別実装から backport)。
# EL 合成は文頭の無音リードインがほぼゼロで 0.03 秒で最大音量に立ち上がるため
# 「頭が切れたよう」に聞こえる。先頭に短い無音パッド + 微小フェードインを入れて自然化。
# 注意: この分だけ各 line が後ろにずれるので、字幕アライメント側(_align)も同じ秒数を参照すること
#       (= HEAD_PAD_SEC を single source 化、字幕は line["start"] + HEAD_PAD_SEC を speech_start にする)。
HEAD_PAD_SEC = 0.12     # 先頭無音パッド(旧 0.05 → 別実装 0.12)
HEAD_FADE_SEC = 0.008   # 立ち上がりフェードイン(EL v3 の急峻な onset を緩和)


def concat_slide_audio(audio_dir: Path, sid: str, audio_paths: list[Path], lines: list[dict]) -> Path:
    import numpy as np
    import soundfile as sf
    out = audio_dir / f"{sid}_concat.wav"
    arrs: list = []
    sr_out = None
    cur = 0.0
    for li, line in enumerate(lines):
        wav = audio_paths[li]
        if not wav.exists():
            print(f"  [A2 skip] {sid}_{li:02d} (wav missing)")
            continue
        a, sr = sf.read(wav)
        if sr_out is None:
            sr_out = sr
        if a.ndim > 1:
            a = a.mean(axis=1)
        a = a.astype(np.float32)
        # 文頭頭切れ対策: 先頭無音パッド + 微小フェードイン(EL v3 の急峻な onset を緩和)
        if len(a) > 0:
            # 音声本体の立ち上がりに HEAD_FADE_SEC の線形フェードインを掛ける
            n_fade = int(sr * HEAD_FADE_SEC)
            if n_fade > 0 and len(a) >= n_fade:
                ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
                a[:n_fade] *= ramp
            head_pad = np.zeros(int(sr * HEAD_PAD_SEC), dtype=np.float32)
            a = np.concatenate([head_pad, a])
        cap = (line.get("caption") or "").strip()
        last = cap[-1] if cap else ""
        # フレーズ間の余韻(言い切った後の無音)。env で上書き可。
        # 別の動画で「フレーズごとの間が詰まりすぎ」指摘 → 引き上げ
        #   (旧: 句点0.25 / 読点0.12 / その他0.08。ゆったりした動画は落ち着いたテンポを優先)
        if last in "。!?！？":
            gap_sec = float(os.environ.get("POST_VIDEO_GAP_END", "0.85"))
        elif last in "、・":
            gap_sec = float(os.environ.get("POST_VIDEO_GAP_COMMA", "0.45"))
        else:
            gap_sec = float(os.environ.get("POST_VIDEO_GAP_OTHER", "0.30"))
        gap = np.zeros(int(sr * gap_sec), dtype=np.float32)
        line["start"] = cur
        line["end"] = cur + len(a) / sr
        line["tail_gap"] = gap_sec
        arrs.append(a)
        arrs.append(gap)
        cur = line["end"] + gap_sec
    arrs.append(np.zeros(int((sr_out or 44100) * 0.7), dtype=np.float32))  # 0.7s tail
    concat = np.concatenate(arrs) if arrs else np.zeros(int((sr_out or 44100) * 1.0), dtype=np.float32)
    sf.write(out, concat, sr_out or 44100)
    return out


# === ASS 字幕(本編と同じロジック)===

PARTICLES = set("をがにでとはも")
PARTICLE_COMPOUND = ["から", "まで", "より", "には", "にも", "では", "でも", "とは"]


def _split_long_chunk(c: str, max_line: int) -> list[str]:
    """長い chunk を max_line 字以内に細分化する。

    v2 改訂(spike-subtitle-boundary): 助詞境界選択を「中央寄り優先」に変更。
    元の実装は max_line から逆順に走査するため末尾寄りの助詞で切れやすく、
    次の entry との境界が助詞になりやすかった。
    中央付近(max_line*0.4 〜 max_line*0.8)のレンジを先に探し、
    なければ末尾寄りへ拡張する。
    """
    if len(c) <= max_line:
        return [c]
    out = []
    cur = c
    while len(cur) > max_line:
        cut = -1
        # 優先ゾーン: 中央寄り(40%-100% of max_line)を前方から探す
        lo = max(max_line // 2, max_line * 4 // 10)
        hi = min(max_line, len(cur))
        # 前方スキャン: より中央に近い位置を優先
        for end in range(lo, hi + 1):
            if end >= 2 and cur[end-2:end] in PARTICLE_COMPOUND:
                cut = end; break
            if end >= 1 and cur[end-1] in PARTICLES and end < len(cur):
                cut = end; break
        if cut == -1:
            cut = max_line
        # 英数字連続中での切断回避 (例: "MLB" を "ML" + "B" に分断する事故対策
        # ffmpeg ass filter が崩れる現象を確認した)。 cut 前後が英数字なら境界 shift。
        if 0 < cut < len(cur) and re.match(r"[A-Za-z0-9]", cur[cut - 1]) and re.match(r"[A-Za-z0-9]", cur[cut]):
            forward = cut
            while forward < len(cur) and re.match(r"[A-Za-z0-9]", cur[forward]):
                forward += 1
            backward = cut
            while backward > 0 and re.match(r"[A-Za-z0-9]", cur[backward - 1]):
                backward -= 1
            # max_line + 5 字猶予で forward 優先 (元 cut に近い方)、 だめなら backward (短いが安全)
            if forward <= max_line + 5:
                cut = forward
            elif backward > 0:
                cut = backward
            # 両方不適は そのまま (英数字でも max_line で切る、 旧挙動踏襲)
        out.append(cur[:cut])
        cur = cur[cut:]
    if cur:
        out.append(cur)
    return out


# 句読点セット(INTER/INTRA の entry 境界として許容される末尾文字)
ENTRY_END_PUNCT = set("。、！？!?,」』）)\"'・")


def split_caption_for_display(caption: str, max_line: int = MAX_LINE, max_lines: int = MAX_LINES) -> list[str]:
    """1 文を字幕エントリに分割。max_lines は厳守(3 行事故の修正済み)。

    分割アルゴリズム(v2、spike-subtitle-boundary):
      1. 句読点で chunks に
      2. **短い chunk(4 字未満)は前 chunk に吸収**(「す。」断片対策、 entry 段階ではなく chunk 段階で防止)
      3. 長い chunk は助詞境界 / 機械分割で細分化(中央寄り優先)
      4. greedy で 1 行 ≤ max_line 字に詰める
      5. **句読点優先 entry 化**: line 列を走査し、
         「現在 entry の最後 line が句読点で終わる」 ならそこで entry を閉じる。
         句読点なしで max_lines 行に達した場合のみ強制終端。
         (max_lines=2 厳守、3 行は絶対作らない)
    """
    s = caption.strip()
    if not s:
        return []
    chunks = re.split(r"(?<=[、。!?!?・」』])", s)
    chunks = [c.strip() for c in chunks if c.strip()] or [s]

    # === 短い chunk を前 chunk に吸収(「す。」「ね。」断片対策)===
    # 4 字未満の chunk は前 chunk に merge。連結後 max_line を超える場合は merge しない
    # (line 詰めで自然に 2 行に分割される)
    # 追加ガード(過去の事故): 前 chunk が句点(「。!?」)で終わってる時は
    # merge しない。 短い chunk を merge すると 「候補A。致命・」 のように句点が
    # chunk 中央に埋まり、 line packing の forced break が効かなくなる。
    MIN_CHUNK = 4
    merged_chunks: list[str] = []
    for c in chunks:
        if (merged_chunks
                and len(c) < MIN_CHUNK
                and len(merged_chunks[-1]) + len(c) <= max_line * max_lines
                and merged_chunks[-1][-1] not in SENTENCE_END):
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
        # 句点境界は強制行切替(過去の事故対策)
        # 「〜ですよね。」 + 「修正は、」 を 1 行に詰めると次文冒頭が貼り付く
        if cur[-1] in SENTENCE_END:
            lines.append(cur); cur = c
        elif len(cur) + len(c) <= max_line:
            cur += c
        else:
            lines.append(cur); cur = c
    if cur:
        lines.append(cur)
    # === 句読点優先 entry 化(v2、spike-subtitle-boundary)===
    # 元実装: range(0, len(lines), max_lines) で機械的に2行ずつまとめる
    # → entry 末尾が助詞で終わるケースが多発(INTER 問題)
    #
    # v2: line 列を走査し、「現在 entry の最後 line が句読点で終わる」 なら
    # そこで entry を閉じる(1 行 entry もあり得る)。
    # 句読点なしで max_lines 行に達した場合のみ強制終端(3 行は絶対作らない)。
    entries: list[str] = []
    entry_lines: list[str] = []
    for i, line in enumerate(lines):
        entry_lines.append(line)
        is_last_line = (i == len(lines) - 1)
        at_max = (len(entry_lines) >= max_lines)
        ends_at_punct = line[-1] in ENTRY_END_PUNCT if line else False
        if is_last_line or at_max or ends_at_punct:
            entries.append("\\N".join(entry_lines))
            entry_lines = []
    if entry_lines:
        # 残余行(通常は発生しないが安全ガード)
        entries.append("\\N".join(entry_lines))

    # === 出口アサーション: 句点跨ぎ防止(過去の事故 構造化)===
    # 各 line 内に「。!?！?」 が末尾以外で出現したら NG(次文冒頭が貼り付いている)
    for entry in entries:
        for line in entry.split("\\N"):
            for idx, ch in enumerate(line):
                if ch in SENTENCE_END and idx < len(line) - 1:
                    raise AssertionError(
                        f"caption line crosses sentence boundary at idx={idx} "
                        f"({ch!r}): {line!r} (full caption={caption!r})"
                    )
    return entries


def _sanitize_segments(segments, min_dur=MIN_DUR, bridge_gap=True):
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s[0])
    out: list[list] = []
    for st, et, txt in sorted_segs:
        if out and st < out[-1][1]:
            st = out[-1][1]
        if et < st:
            et = st
        out.append([st, et, txt])
    if bridge_gap and len(out) > 1:
        for i in range(len(out) - 1):
            # 終了時刻を常に「次 entry 開始 - 10ms」 に強制
            # 元実装 (条件 < 発火) では「終了 = 次開始」 重畳 case を逃していた
            # force_align は entry 境界をピッタリ配置するので強制が必要(fix v2)
            target_end = out[i + 1][0] - 0.01
            if target_end < out[i][0]:
                target_end = out[i][0]
            out[i][1] = target_end
    merged: list[list] = []
    for s in out:
        st, et, txt = s
        if et - st < min_dur:
            if merged:
                merged[-1][1] = max(merged[-1][1], st + min_dur)
            else:
                et = st + min_dur
                merged.append([st, et, txt])
            continue
        # ★FIX: 短 entry マージで前エントリ end が伸長した後、
        # 次エントリ start が新 end より前になる場合の start ずれを修正
        # libass collision avoidance により字幕が画面外に押し出される問題を防ぐ
        if merged and st < merged[-1][1]:
            st = merged[-1][1]
            if et < st:
                et = st
            if et - st < min_dur:
                if merged:
                    merged[-1][1] = max(merged[-1][1], st + min_dur)
                continue
        merged.append([st, et, txt])
    return [(a, b, c) for a, b, c in merged]


def write_ass(segments, ass_path: Path, play_w=VIDEO_W, play_h=VIDEO_H,
              fontname=FONT_NAME, fontsize=FONT_SIZE, outline=OUTLINE,
              band_top=SLIDE_H, band_bottom=VIDEO_H):
    # === 字幕位置: ハードコード固定(再発防止) ===
    # Alignment=8 (上中央) + MarginV 固定値 で字幕位置を制御
    # 画面上 0-880px=スライド領域、880-1080px=字幕領域(高さ200px)
    # 関係式: y_min = MarginV + 13  (Alignment=8, filter_complex 再現実測)
    #   MarginV=897: y_min=910 (スライド直下+30px余裕), y_max=1045 (下端35px余裕)
    # 変更時は以下を同期更新すること:
    #   - content_pipeline/scripts/qa_subtitle_boundary.py の期待値チェック
    #   - tests/test_post_video_subtitle_position.py の期待値
    SUBTITLE_ALIGNMENT = 8   # 上中央(変更禁止)
    SUBTITLE_MARGIN_V = 897  # 全行数共通: y_min=910(上余白30px)、y_max=1045(下余裕35px)

    def fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = t - h*3600 - m*60
        return f"{h:01d}:{m:02d}:{s:05.2f}"

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_w}\nPlayResY: {play_h}\n"
        "ScaledBorderAndShadow: no\n"
        f"WrapStyle: {WRAP_STYLE}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fontname},{fontsize},&H00FFFFFF,&H00000000,1,{outline},0,{SUBTITLE_ALIGNMENT},40,40,{SUBTITLE_MARGIN_V},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    sanitized = _sanitize_segments(segments)
    body_lines = []
    for st, et, txt in sanitized:
        body_lines.append(f"Dialogue: 0,{fmt(st)},{fmt(et)},Default,,0,0,{SUBTITLE_MARGIN_V},,{txt}")
    ass_path.write_text(header + "\n".join(body_lines) + "\n", encoding="utf-8")


# === per-slide clip ===

def _file_sha_short(p: Path) -> str:
    """ファイルの sha256 16 文字 prefix。 shared→per-post 同期 safety net 用。"""
    import hashlib
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ensure_shared_synced(slide: dict, per_post_anim: Path) -> None:
    """compose 直前の safety net: shared slide の anim/audio が per-post copy と
    content_hash で一致しているか確認、 不一致なら強制 copy + WARN ログ。

    parallel_render を呼び忘れた / 古いキャッシュが残った状況で
    「stale per-post anim が compose に渡る」 事故を防ぐ最後の防御線
    (outro 3秒問題 由来)。
    """
    if not slide.get("shared_id"):
        return
    sid = slide["id"]
    # anim
    shared_anim = slide.get("anim_path")
    if shared_anim and shared_anim.exists():
        src_sha = _file_sha_short(shared_anim)
        dst_sha = _file_sha_short(per_post_anim)
        if not per_post_anim.exists() or src_sha != dst_sha:
            per_post_anim.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shared_anim, per_post_anim)
            print(
                f"  [SAFETY] shared anim refreshed for {sid} ({slide['shared_id']}): "
                f"src={src_sha} dst={dst_sha} (parallel_render skipped?)",
                flush=True,
            )
    # audio (shared wav は paths.audio_dir(topic)/<sid>_<li>.wav に copy 済 想定)
    for li, src_wav in enumerate(slide.get("audio_paths", [])):
        # shared 側 src は shared_slide_dir/line_<li>.wav
        from . import paths as _paths
        shared_wav = _paths.shared_slide_dir(slide["shared_id"]) / f"line_{li:02d}.wav"
        # per-post 側 dst は audio_dir/<sid>_<li>.wav
        per_post_wav = _paths.audio_dir(slide["__topic__"]) / f"{sid}_{li:02d}.wav" if "__topic__" in slide else None
        if not shared_wav.exists() or per_post_wav is None:
            continue
        if not per_post_wav.exists() or _file_sha_short(shared_wav) != _file_sha_short(per_post_wav):
            per_post_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shared_wav, per_post_wav)
            print(
                f"  [SAFETY] shared wav refreshed for {sid}_{li:02d}",
                flush=True,
            )


def build_clip(topic: str, slide: dict, audio_path: Path, with_subtitle: bool = True) -> Path:
    """per-slide clip mp4 を生成。

    with_subtitle=False の場合は字幕 ASS の焼き込みをスキップ(無音/字幕無しの clip)。
    ASS 自体は work/_tmp_{sid}.ass に書き出されるので、 後段 compose_final.py で
    global ASS にマージして burn できる。 詳細は src/post_video/compose_final.py。
    """
    sid = slide["id"]
    clip_out = paths.clips_dir(topic) / f"{sid}.mp4"
    anim_mp4 = paths.anim_dir(topic) / f"{sid}.mp4"
    raw_fallback = slide["raw_path"]

    # safety net: shared slide の anim/audio が stale な場合に強制同期
    # (parallel_render が呼ばれていない / mtime 比較の旧 cache がある等の防御)
    slide_with_topic = {**slide, "__topic__": topic}
    _ensure_shared_synced(slide_with_topic, anim_mp4)

    # 字幕 segments — fallback chain:
    #   1次: ElevenLabs convert_with_timestamps の char-level ground truth (el_align)
    #   2次: VAD silence + 決定論的 gap (vad_align、 現状未呼出だが インフラ あり)
    #   3次: whisperx wav2vec2 char-level forced alignment (force_align、 現行)
    #   4次: 比例分割
    # 1 次 el_align を導入。 timestamps.json 不在 line は 自動的に 3 次に降格。
    from . import force_align, vad_align, el_align
    import soundfile as _sf
    segments: list = []
    for li, line in enumerate(slide["lines"]):
        cap = line["caption"]
        entries = split_caption_for_display(cap)
        if not entries:
            continue
        # HEAD_PAD_SEC はモジュール定数(concat_slide_audio と single source、現 0.12s)を参照
        if len(entries) == 1:
            # 単 entry も 1 次 (el_align) を 試行 (ElevenLabs timestamps の char[0..n-1] 全体を 使う)
            line_wav = slide["audio_paths"][li] if li < len(slide["audio_paths"]) else None
            from . import synth as _synth
            _kana_dict = _synth._load_kana_dict()
            entries_kana = [_synth._apply_kana_dict(entries[0], _kana_dict)]
            speech_start = line["start"] + HEAD_PAD_SEC
            el_timings = None
            if line_wav and line_wav.exists():
                ts = el_align.load_timestamps(paths.audio_dir(topic), sid, li)
                if ts is not None:
                    el_timings = el_align.align_entries_from_timestamps(
                        entries_kana, entries, ts,
                        line_start_offset=speech_start,
                        wav_dur=line["end"] - speech_start,
                    )
            if el_timings:
                st, et = el_timings[0]
                segments.append((st, et, entries[0]))
            else:
                segments.append((speech_start, line["end"], entries[0]))
        else:
            # 複数 entries — fallback chain で alignment 試行
            line_wav = slide["audio_paths"][li] if li < len(slide["audio_paths"]) else None
            words: list = []
            vad_timings: list | None = None
            el_timings: list | None = None
            # caption に kana_dict 適用 (alignment 用、 表示用 caption は entries そのまま)
            from . import synth as _synth
            _kana_dict = _synth._load_kana_dict()
            entries_kana = [_synth._apply_kana_dict(e, _kana_dict) for e in entries]
            # === 1 次: ElevenLabs char-level timestamps (ground truth ±1-5ms) ===
            if line_wav and line_wav.exists():
                ts = el_align.load_timestamps(paths.audio_dir(topic), sid, li)
                if ts is not None:
                    # wav_dur で trim 越え clamp
                    try:
                        wav_dur_only = line["end"] - line["start"] - HEAD_PAD_SEC
                    except Exception:
                        wav_dur_only = None
                    el_timings = el_align.align_entries_from_timestamps(
                        entries_kana, entries, ts,
                        line_start_offset=line["start"] + HEAD_PAD_SEC,
                        wav_dur=wav_dur_only,
                    )
                    if el_timings is None:
                        print(f"  [el_align skip] {sid}_{li:02d}: char alignment mismatch → whisperx fallback")
            # === 3 次: whisperx (el_align 失敗時) ===
            if el_timings is None and line_wav and line_wav.exists():
                try:
                    words = force_align.transcribe(
                        line_wav, language="ja", initial_prompt=_synth._apply_kana_dict(cap, _kana_dict),
                    )
                except Exception as e:
                    print(f"  [align WARN] {sid}_{li:02d}: {e}")
            # fix v2 (過去の事故由来、 user 体感「字幕がかなり早い」):
            # 字幕の onset を 音声 onset に近づけるための補正:
            #   HEAD_PAD_SEC(現 0.12s = concat_slide_audio が各 line wav 前に挿入する無音)だけ
            #   speech_start を後ろにずらす。concat 側と同じ定数を参照するので常に同期する。
            # v5: whisperx wav2vec2 char-level alignment 切替で word.start の systematic
            #   bias がほぼ消えたため、追加の TTS_ONSET_PAD 過補正は 0 に戻した(字幕遅延体感の解消)。
            # head_pad を 0.05→0.12s に延長(別実装、文頭頭切れ対策)。
            #   HEAD_PAD_SEC を single source 化したので、ここは定数追従で自動同期する。
            TTS_ONSET_PAD = 0.0
            speech_start = line["start"] + HEAD_PAD_SEC + TTS_ONSET_PAD
            line_dur = line["end"] - line["start"] - HEAD_PAD_SEC  # speech only (TTS_ONSET_PAD は subtract しない: words 検出範囲は line_wav 全体なので)
            try:
                if el_timings is not None:
                    # 1 次 el_align (ElevenLabs ground truth) を そのまま 採用
                    timings = el_timings
                else:
                    # 3 次 whisperx: entries_kana を渡して transcript と char-level 整合
                    timings = force_align.align_entries_to_wav(
                        entries_kana, words,
                        line_start_offset=speech_start,
                        line_duration=line_dur,
                    )
            except Exception as e:
                print(f"  [align fallback] {sid}_{li:02d}: {e} → 比例分割")
                # G-Audio: fallback 件数を qa_post_video が拾うため、
                # topic/work/_force_align_fallback.log に追記する
                try:
                    log_path = paths.work_dir(topic) / "_force_align_fallback.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{sid}\t{li:02d}\t{type(e).__name__}: {e}\n")
                except Exception:
                    pass
                chunk = line_dur / len(entries)
                # fallback 比例分割も speech 開始基準 (head_pad ぶん offset 込み)
                timings = [(speech_start + i*chunk, speech_start + (i+1)*chunk) for i in range(len(entries))]
            # 字幕ラグ対策(onset補正対応):
            # 先頭 entry は speech_start (= line["start"] + HEAD_PAD_SEC + TTS_ONSET_PAD) にスナップ。
            # 全 entry の onset 補正と整合させる(旧 line["start"] + HEAD_PAD_SEC では 50ms 早かった)。
            if timings:
                _, first_et = timings[0]
                snapped_st = speech_start
                if first_et > snapped_st:
                    timings[0] = (snapped_st, first_et)
            for ent, (st, et) in zip(entries, timings):
                segments.append((st, et, ent))
    if segments:
        slide_end_sec = max(line["end"] for line in slide["lines"]) + 0.7
        # 最終字幕は slide 終端の 0.10s 手前で止める(フェードアウト中の字幕残留を防ぐ)
        last_subtitle_end = slide_end_sec - 0.10  # was slide_end_sec のまま延長
        segs_sorted = sorted(segments, key=lambda s: s[0])
        last_st, last_et, last_txt = segs_sorted[-1]
        segs_sorted[-1] = (last_st, max(last_et, last_subtitle_end), last_txt)
        # 重畳強制解消: 隣接 entry の end が次 start を超えたら 10ms 手前で止める
        # (write_ass 内 _sanitize_segments が効かない経路の保険 fix v3)
        for i in range(len(segs_sorted) - 1):
            st_i, et_i, txt_i = segs_sorted[i]
            st_next = segs_sorted[i + 1][0]
            target_end = st_next - 0.01
            if et_i > target_end and target_end > st_i:
                segs_sorted[i] = (st_i, target_end, txt_i)
        segments = segs_sorted

    ass_path = paths.work_dir(topic) / f"_tmp_{sid}.ass"
    write_ass(segments, ass_path)
    ass_escaped = str(ass_path).replace("'", "\\'").replace(":", "\\:")

    # option A: per-clip a/v drift 根治
    # 旧来 `-shortest` だけだと h264_videotoolbox が 25fps 境界で video を丸めて
    # video_dur が audio_dur より 30-40ms 短くなる。 これが 十数 clip の concat で数百 ms
    # 累積し、 字幕/音声 drift として知覚される (過去の事故由来)。
    #
    # 対策: audio_path を ffprobe して実 audio_dur を取得 → 1/25 秒境界に切り上げて
    # video_target_dur に snap。 video は filter chain 末尾 tpad で必ず target に届く
    # ように padding。 audio は apad で同 target に extend。 `-shortest` 撤去、 `-t`
    # で両ストリームを target_dur で固定 → 出力 container で v_dur == a_dur となる。
    FPS = 25.0
    audio_dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        text=True,
    ).strip())
    target_dur = math.ceil(audio_dur * FPS) / FPS  # frame 境界に切り上げ
    duration = target_dur  # 旧 max(line.end)+0.7 ではなく audio 実 duration を使う

    # filter_complex: 字幕焼き込みは with_subtitle=True 時のみ最後のステップで適用。
    # with_subtitle=False なら [withslide] をそのまま [v] にする(字幕は compose_final.py
    # で global ASS に統合して 最終 mp4 にだけ焼く 新パイプライン)。
    # tpad で video output を必ず target_dur に届くようにする (anim が短くても OK)
    # filter chain: overlay → [withslide] → (ass → [vass]) → tpad → [v]
    # 旧コードは overlay_label が with_subtitle 時に "[vass]" になっており、 ass の input "[withslide]"
    # が未定義 + ass output "[vass]" がラベル重複で unconnected エラーになる構造バグがあった。
    # 過去 build は clip.mp4 cache hit で実 compose が走らず顕在化していなかった
    # (cache 全 invalidate 時に全 slide が fail して発覚)。
    subtitle_step = f";[withslide]ass='{ass_escaped}'[vass]" if with_subtitle else ""
    overlay_label = "[withslide]"  # 常に [withslide] 名で chain 化
    pad_input = "[vass]" if with_subtitle else "[withslide]"
    # 最終 stage: tpad で末尾フレーム clone し target_dur まで延長 → fps で CFR 25
    tpad_step = f";{pad_input}tpad=stop_mode=clone:stop_duration={target_dur},fps={FPS:g}[v]"
    # blob anim mp4 があれば動画入力、なければ静止画
    if anim_mp4.exists():
        filter_complex = (
            f"color=c=black:s={VIDEO_W}x{VIDEO_H}:d={target_dur}[bg];"
            f"[0:v]scale={VIDEO_W}:{SLIDE_H}:force_original_aspect_ratio=decrease,"
            f"pad={VIDEO_W}:{SLIDE_H}:(ow-iw)/2:(oh-ih)/2:white[s];"
            f"[bg][s]overlay=0:0{overlay_label}"
            f"{subtitle_step}"
            f"{tpad_step}"
        )
        video_input = ["-i", str(anim_mp4)]
    else:
        print(
            f"WARN: {sid} anim.mp4 不在 → 静止画フォールバック。"
            f" parallel_render が正常完了していれば発生しないはず。",
            flush=True,
        )
        filter_complex = (
            f"color=c=black:s={VIDEO_W}x{VIDEO_H}:d={target_dur}[bg];"
            f"[0:v]scale={VIDEO_W}:{SLIDE_H}:force_original_aspect_ratio=decrease,"
            f"pad={VIDEO_W}:{SLIDE_H}:(ow-iw)/2:(oh-ih)/2:white[s];"
            f"[bg][s]overlay=0:0{overlay_label}"
            f"{subtitle_step}"
            f"{tpad_step}"
        )
        video_input = ["-loop", "1", "-i", str(raw_fallback)]

    # 音声側 filter: apad で target_dur まで無音 padding
    audio_filter = f"apad=whole_dur={target_dur}"

    cmd = [
        "ffmpeg", "-y", *video_input, "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-af", audio_filter,
        "-map", "[v]", "-map", "1:a",
        "-c:v", "h264_videotoolbox", "-b:v", "4M", "-pix_fmt", "yuv420p",
        "-r", f"{FPS:g}", "-vsync", "cfr",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{target_dur}",  # -shortest 撤去 (option A: a/v 両方 target_dur で揃える)
        str(clip_out),
    ]
    subtitle_flag = "+subtitle焼" if with_subtitle else "−subtitle (compose_final 用)"
    print(f"  [C] {sid} compose ({duration:.2f}s, {len(segments)} subtitles, {subtitle_flag})")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # 失敗 cmd 全文を dump (再現確認用)
        import shlex
        sys.stderr.write(f"\n[debug] failed cmd (sid={sid}):\n{shlex.join(cmd)}\n\n")
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


def _build_clip_worker(args: tuple) -> tuple[str, Path | None, str]:
    """ProcessPool 用 worker。 (sid, clip_path, msg) を返す。

    args = (topic, slide, audio_path, with_subtitle).
    """
    topic, slide, audio_path, with_subtitle = args
    try:
        clip = build_clip(topic, slide, audio_path, with_subtitle=with_subtitle)
        return (slide["id"], clip, "OK")
    except Exception as e:
        return (slide["id"], None, f"{type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--workers", type=int, default=3,
                    help="per-slide build_clip 並列数(default 3、 force_align 重い)")
    ap.add_argument("--no-subtitle", action="store_true",
                    help="per-clip mp4 に字幕を焼かない (subtitle as artifact pattern)。 "
                         "ASS は生成、 後段 compose_final.py で global ASS にまとめて最終 mp4 に焼く。 "
                         "UI overlay 編集の WYSIWYG を真にするため ")
    ap.add_argument("--only-slides", default=None,
                    help="カンマ区切り sid。 指定 slide のみ clip 再生成、 他 sid は skip。 "
                         "UI per-scene clip regen 用 ")
    args = ap.parse_args(argv)

    topic = args.topic
    with_subtitle = not args.no_subtitle
    paths.audio_dir(topic).mkdir(parents=True, exist_ok=True)
    paths.clips_dir(topic).mkdir(parents=True, exist_ok=True)

    # G-Audio: 旧 force_align fallback ログを除去(今回 build の fallback 件数のみ反映)
    fallback_log = paths.work_dir(topic) / "_force_align_fallback.log"
    if fallback_log.exists():
        fallback_log.unlink()

    slides = load_slides(topic)

    only: set[str] | None = None
    if args.only_slides:
        only = set()
        for t in args.only_slides.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                only.add(f"{int(t):02d}")
            except ValueError:
                print(f"[--only-slides] WARN: invalid sid {t!r} 無視")
        if only:
            before = len(slides)
            slides = [s for s in slides if s["id"] in only]
            print(f"[--only-slides] {sorted(only)} のみ ({before} → {len(slides)} slides)")
            if not slides:
                print("  (該当 slide 無し、 何もせず終了)")
                return 0

    print("=== Phase A2: 各 slide 内の wav 結合 ===")
    slide_concat: dict[str, Path] = {}
    for s in slides:
        sid = s["id"]
        if not all(p.exists() for p in s["audio_paths"]):
            print(f"  [A2 skip] {sid}: 一部 wav 不在(synth.py 先に実行)")
            continue
        slide_concat[sid] = concat_slide_audio(paths.audio_dir(topic), sid, s["audio_paths"], s["lines"])

    subtitle_mode = "字幕焼き込み (旧パイプ)" if with_subtitle else "字幕無し (compose_final で最終焼き込み)"
    print(f"\n=== Phase C: per-slide clip 生成(workers={args.workers}, {subtitle_mode}) ===")
    work_args = [(topic, s, slide_concat[s["id"]], with_subtitle)
                 for s in slides if s["id"] in slide_concat]
    clip_paths_dict: dict[str, Path] = {}
    n_fail = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_build_clip_worker, a): a[1]["id"] for a in work_args}
        for f in as_completed(futures):
            sid, clip, msg = f.result()
            if clip is not None:
                clip_paths_dict[sid] = clip
                print(f"  [C ok]    {sid}", flush=True)
            else:
                print(f"  [C error] {sid}: {msg}", flush=True)
                n_fail += 1
    if n_fail > 0:
        return 1
    # slide 順に並べる
    clip_paths = [clip_paths_dict[s["id"]] for s in slides if s["id"] in clip_paths_dict]

    # build モード メタ (UI / compose_final 用): subtitle_burned 状態を記録
    from datetime import datetime
    meta = {
        "subtitle_burned": with_subtitle,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "clip_count": len(clip_paths),
    }
    meta_path = paths.clips_dir(topic) / "_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[meta] {meta_path.name}: subtitle_burned={with_subtitle}")

    print("\n=== Phase D: concat ===")
    concat_clips(topic, clip_paths)
    if not with_subtitle:
        print("\n※ 字幕は per-clip に焼かれていません。 final mp4 を視聴/配信する前に "
              "`python -m src.post_video.compose_final --topic {topic}` を実行して字幕を焼き込んでください。"
              .format(topic=topic))
    print("\n=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
