#!/usr/bin/env python3
"""TTS で生成した wav と元テキストから、字幕用の word-level タイムスタンプを取得する。

TTS provider 非依存(SBV2 / ElevenLabs / VOICEVOX どれでも入力 wav として動く)。
faster-whisper で word_timestamps=True を取り、字幕エントリを wav 時間に対応付ける。
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional

_model = None
_model_size: Optional[str] = None


def get_model(size: str = "large-v3", compute_type: str = "int8"):
    """faster-whisper モデルを singleton で取得。
    - 本編 src/qa_transcribe.py に揃え large-v3 を既定(精度優先、初回 ~3GB DL)
    - GPU 非対応の環境では CPU + int8 が安定
    """
    global _model, _model_size
    if _model is None or _model_size != size:
        from faster_whisper import WhisperModel
        _model = WhisperModel(size, device="cpu", compute_type=compute_type)
        _model_size = size
    return _model


def _wav_duration(wav_path: Path) -> float:
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return float(info.duration)
    except Exception:
        return 0.0


def transcribe(
    wav_path: Path,
    language: str = "ja",
    initial_prompt: Optional[str] = None,
    model_size: str = "large-v3",
) -> list[dict]:
    """wav を ASR + whisperx wav2vec2 char-level forced alignment で 文字単位タイミング取得。

    旧実装は faster-whisper word_timestamps を 直接使い、 word.start が 実音 onset より
    50-100ms 早めの bias を持つ + char interpolation が break するため per-entry ±1.6s 誤差
     が発生。 char-level forced align に切替。

    返却 dict は 旧 word-level 互換シグネチャ (word/start/end):
      [{"word": "ク", "start": 0.031, "end": 0.111}, {"word": "ロ", ...}, ...]
    1 char = 1 entry に分解される (caller の align_entries_to_wav は char_times 経由で
    使うので word vs char の違いは透過)。
    """
    wav_dur = _wav_duration(wav_path)
    try:
        import whisperx
        # ASR (faster-whisper backend) で transcript を 取得
        asr_model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
        audio = whisperx.load_audio(str(wav_path))
        result = asr_model.transcribe(audio, language=language, batch_size=4)
        # wav2vec2 forced alignment で char-level timing
        global _align_model, _align_metadata, _align_lang
        if "_align_model" not in globals() or _align_lang != language:
            _align_model, _align_metadata = whisperx.load_align_model(language_code=language, device="cpu")
            _align_lang = language
        aligned = whisperx.align(result["segments"], _align_model, _align_metadata, audio,
                                  device="cpu", return_char_alignments=True)
        words: list[dict] = []
        for seg in aligned.get("segments", []):
            for c in seg.get("chars", []):
                ch = c.get("char", "").strip()
                if not ch:
                    continue
                st = c.get("start")
                et = c.get("end")
                if st is None or et is None:
                    continue
                st = float(st); et = float(et)
                if wav_dur > 0:
                    st = min(st, wav_dur)
                    et = min(et, wav_dur)
                if et < st:
                    et = st
                words.append({"word": ch, "start": st, "end": et})
        words.sort(key=lambda x: x["start"])
        return words
    except Exception as e:
        # whisperx 失敗時は faster-whisper word_timestamps に fallback
        import sys as _sys
        print(f"  [force_align fallback] whisperx 失敗 → faster-whisper word level: {e}", file=_sys.stderr)
        model = get_model(model_size)
        kwargs = {
            "language": language, "word_timestamps": True, "vad_filter": True,
            "condition_on_previous_text": False, "beam_size": 5,
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, _info = model.transcribe(str(wav_path), **kwargs)
        words: list[dict] = []
        for seg in segments:
            if not seg.words:
                continue
            for w in seg.words:
                st = float(w.start); et = float(w.end)
                if wav_dur > 0:
                    st = min(st, wav_dur); et = min(et, wav_dur)
                if et < st: et = st
                if not (w.word or "").strip(): continue
                words.append({"word": w.word, "start": st, "end": et})
        words.sort(key=lambda x: x["start"])
        return words


def _proportional_fallback(
    captions_clean: list[str],
    line_start_offset: float,
    line_duration: Optional[float],
) -> list[tuple[float, float]]:
    """ASR 失敗時:caption の文字数比例で line_duration を分配する。
    均等分割よりは長文と短文の差が反映される。
    """
    if not captions_clean:
        return []
    if line_duration is None or line_duration <= 0:
        return [(line_start_offset, line_start_offset)] * len(captions_clean)
    total_chars = sum(len(c) for c in captions_clean) or 1
    out: list[tuple[float, float]] = []
    cur = 0.0
    for c in captions_clean:
        ratio = len(c) / total_chars
        seg_dur = line_duration * ratio
        out.append((line_start_offset + cur, line_start_offset + cur + seg_dur))
        cur += seg_dur
    return out


def _build_char_times(words: list[dict]) -> list[float]:
    """transcript を char 単位に展開し、各 char の時刻を線形補間で求める。"""
    out: list[float] = []
    for w in words:
        text = w["word"]
        n = len(text)
        if n == 0:
            continue
        dur = max(w["end"] - w["start"], 1e-6)
        for k in range(n):
            out.append(w["start"] + dur * (k / n))
    if words:
        out.append(words[-1]["end"])  # 末端 sentinel
    return out


def align_entries_to_wav(
    caption_entries: list[str],
    words: list[dict],
    line_start_offset: float = 0.0,
    line_duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    """caption_entries の各エントリの (start_sec, end_sec) を wav 上で推定する。

    line_start_offset: スライド全体内の本 line(文)開始時刻(concat 後絶対時刻)
    line_duration: ASR 失敗時の均等分割フォールバック用

    アルゴリズム:
      1. caption 全文(\\N 抜き)と whisper transcript を char 列にする
      2. difflib で 2 つの char 列の matching block を取得
      3. caption の各 char idx に対応する transcript char idx を引く
      4. transcript char idx → 時刻(線形補間)
      5. エントリの開始/終了 char idx をそれぞれ時刻化
    """
    n_entries = len(caption_entries)
    if n_entries == 0:
        return []

    # フォールバック:whisper 失敗 → line duration を均等分割
    if not words:
        if line_duration is None or line_duration <= 0:
            return [(line_start_offset, line_start_offset)] * n_entries
        seg = line_duration / n_entries
        return [
            (line_start_offset + i * seg, line_start_offset + (i + 1) * seg)
            for i in range(n_entries)
        ]

    captions_clean = [e.replace("\\N", "") for e in caption_entries]
    full_caption = "".join(captions_clean)
    transcript = "".join(w["word"] for w in words)
    char_times = _build_char_times(words)

    if not full_caption or not transcript or not char_times:
        # フォールバック
        return _proportional_fallback(captions_clean, line_start_offset, line_duration)

    # caption ↔ transcript の char alignment
    sm = difflib.SequenceMatcher(None, full_caption, transcript)
    matching = sm.get_matching_blocks()
    cap_to_trans: dict[int, int] = {}
    matched_chars = 0
    for m in matching:
        matched_chars += m.size
        for k in range(m.size):
            cap_to_trans[m.a + k] = m.b + k

    # マッチング率が低い → ASR がテキストと大きく違う → フォールバック
    match_ratio = matched_chars / max(len(full_caption), 1)
    if match_ratio < 0.5:
        return _proportional_fallback(captions_clean, line_start_offset, line_duration)

    sorted_keys = sorted(cap_to_trans.keys())

    def cap_idx_to_time(cap_idx: int) -> float:
        # 完全一致点
        if cap_idx in cap_to_trans:
            ti = min(cap_to_trans[cap_idx], len(char_times) - 1)
            return char_times[ti]
        # 補間
        prev_k = None
        nxt_k = None
        # binary-ish: sorted_keys から近傍を探す(線形でも OK、サイズ小)
        for k in sorted_keys:
            if k <= cap_idx:
                prev_k = k
            elif k > cap_idx:
                nxt_k = k
                break
        if prev_k is None and nxt_k is None:
            return 0.0
        if prev_k is None:
            ti = min(cap_to_trans[nxt_k], len(char_times) - 1)
            return char_times[ti]
        if nxt_k is None:
            ti = min(cap_to_trans[prev_k], len(char_times) - 1)
            return char_times[ti]
        prev_t = char_times[min(cap_to_trans[prev_k], len(char_times) - 1)]
        nxt_t = char_times[min(cap_to_trans[nxt_k], len(char_times) - 1)]
        ratio = (cap_idx - prev_k) / max(nxt_k - prev_k, 1)
        return prev_t + (nxt_t - prev_t) * ratio

    # エントリごとの (start, end) を計算(line wav 内相対時刻)
    line_dur_cap = line_duration if line_duration and line_duration > 0 else (words[-1]["end"] if words else 0.0)
    timings_rel: list[tuple[float, float]] = []
    cur = 0
    prev_end = 0.0
    for entry in captions_clean:
        st = cap_idx_to_time(cur)
        cur += len(entry)
        et = cap_idx_to_time(cur) if cur < len(full_caption) else (words[-1]["end"] if words else st)
        # サニティ:line duration を超えない
        st = min(max(st, 0.0), line_dur_cap)
        et = min(max(et, st), line_dur_cap)
        # 単調増加保証
        if st < prev_end:
            st = prev_end
        if et <= st:
            et = min(st + 0.1, line_dur_cap)
        prev_end = et
        timings_rel.append((st, et))

    # フォールバック判定:全エントリの合計 duration が line_duration の半分以下なら破綻
    total_span = sum(et - st for st, et in timings_rel)
    if line_duration and line_duration > 0 and total_span < line_duration * 0.4:
        return _proportional_fallback(captions_clean, line_start_offset, line_duration)

    # 異常検知: timings_rel の各 entry duration が line 比例分割の 30% 未満なら破綻
    # 過去に long caption の word alignment 逆順により duration 0.1s に
    # 縮んで _sanitize_segments で merge 消失する事象を防ぐ(fix)
    if line_duration and line_duration > 0:
        expected_dur = line_duration / n_entries
        min_acceptable = expected_dur * 0.3  # ハードコード閾値
        for st, et in timings_rel:
            if (et - st) < min_acceptable:
                return _proportional_fallback(captions_clean, line_start_offset, line_duration)

    return [(line_start_offset + s, line_start_offset + e) for s, e in timings_rel]


if __name__ == "__main__":
    # 動作確認:1 wav に対して transcribe + align
    import json
    import sys

    if len(sys.argv) < 3:
        print("usage: force_align.py <wav> <expected_text>")
        sys.exit(1)
    wav = Path(sys.argv[1])
    text = sys.argv[2]
    words = transcribe(wav, initial_prompt=text)
    print(json.dumps(words, ensure_ascii=False, indent=2))
